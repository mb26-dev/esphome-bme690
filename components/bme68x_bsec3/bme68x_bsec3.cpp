/**
 * ESPHome component for BME688/BME690 air quality sensor via Bosch BSEC3.
 *
 * Runs the BSEC3 algorithm in a dedicated FreeRTOS task. Sensor readings
 * are passed to BSEC which produces IAQ, CO2eq, VOC, and other derived
 * outputs. State is saved to NVS for calibration persistence:
 *   - periodically (state_save_interval), but only while IAQ accuracy is 3,
 *     so a degraded state never overwrites a good calibration;
 *   - on safe shutdown (OTA update, restart button, API/WiFi reboot timeout).
 * While BSEC is stabilizing (accuracy 0, ~5 min after boot) the IAQ-derived
 * values are published as unknown instead of a misleading ramp from 0.
 *
 * Patch: all large BSEC work/config/state buffers live on the heap instead of
 * the (small) ESPHome loop-task stack; this avoids stack overflow at boot.
 */

#include "esphome/core/defines.h"
#include "bme68x_bsec3.h"
#include "esphome/core/log.h"
#include "esphome/core/preferences.h"

#include "esp_timer.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

namespace esphome {
namespace bme68x_bsec3 {

static const char *const TAG = "bme68x_bsec3";

static const uint32_t BSEC_TASK_STACK_SIZE = 12288;
static const uint32_t BSEC_TASK_PRIORITY = 5;
static const uint32_t BSEC_TASK_STARTUP_DELAY_MS = 5000;
static const uint32_t BSEC_SHUTDOWN_LOCK_TIMEOUT_MS = 3000;

// NVS key for BSEC state persistence
static const uint32_t BSEC_STATE_HASH = 0xB5EC0003;

// --- BME69x I2C Callbacks ---

BME69X_INTF_RET_TYPE BME68xBSEC3Component::i2c_read_(uint8_t reg_addr, uint8_t *data, uint32_t len, void *intf_ptr) {
  auto *self = static_cast<BME68xBSEC3Component *>(intf_ptr);
  auto err = self->read_register(reg_addr, data, len);
  return err == i2c::ERROR_OK ? BME69X_OK : BME69X_E_COM_FAIL;
}

BME69X_INTF_RET_TYPE BME68xBSEC3Component::i2c_write_(uint8_t reg_addr, const uint8_t *data, uint32_t len, void *intf_ptr) {
  auto *self = static_cast<BME68xBSEC3Component *>(intf_ptr);
  auto err = self->write_register(reg_addr, data, len);
  return err == i2c::ERROR_OK ? BME69X_OK : BME69X_E_COM_FAIL;
}

void BME68xBSEC3Component::delay_us_(uint32_t period, void *intf_ptr) {
  if (period >= 1000) {
    vTaskDelay(pdMS_TO_TICKS(period / 1000));
  } else {
    esp_rom_delay_us(period);
  }
}

// --- Component Lifecycle ---

float BME68xBSEC3Component::get_setup_priority() const {
  return setup_priority::DATA;
}

void BME68xBSEC3Component::setup() {
  ESP_LOGCONFIG(TAG, "Setting up BME68x BSEC3...");

  // Create mutexes for shared sensor data and for the BSEC library itself
  this->data_mutex_ = xSemaphoreCreateMutex();
  this->bsec_mutex_ = xSemaphoreCreateMutex();
  if (this->data_mutex_ == nullptr || this->bsec_mutex_ == nullptr) {
    this->mark_failed();
    ESP_LOGE(TAG, "Failed to create mutex");
    return;
  }

  // Initialize BME69x device struct (zeroed first so no stray function pointers)
  memset(&this->bme69x_dev_, 0, sizeof(this->bme69x_dev_));
  this->bme69x_dev_.intf = BME69X_I2C_INTF;
  this->bme69x_dev_.read = i2c_read_;
  this->bme69x_dev_.write = i2c_write_;
  this->bme69x_dev_.delay_us = delay_us_;
  this->bme69x_dev_.intf_ptr = this;
  this->bme69x_dev_.amb_temp = 25;  // Initial ambient temp estimate

  // Initialize BME69x sensor
  int8_t bme_status = bme69x_init(&this->bme69x_dev_);
  if (bme_status != BME69X_OK) {
    this->mark_failed();
    ESP_LOGE(TAG, "BME69x init failed: %d", bme_status);
    return;
  }
  ESP_LOGI(TAG, "BME69x chip ID: 0x%02X", this->bme69x_dev_.chip_id);

  // Allocate BSEC3 instance memory
  size_t instance_size = bsec_get_instance_size();
  this->bsec_instance_ = (uint8_t *) malloc(instance_size);
  if (this->bsec_instance_ == nullptr) {
    this->mark_failed();
    ESP_LOGE(TAG, "Failed to allocate BSEC instance (%u bytes)", (unsigned) instance_size);
    return;
  }
  memset(this->bsec_instance_, 0, instance_size);
  ESP_LOGI(TAG, "BSEC instance allocated: %u bytes", (unsigned) instance_size);

  // Initialize BSEC3 instance
  this->init_bsec_();
  if (this->is_failed()) {
    return;
  }

  // Launch dedicated FreeRTOS task for BSEC processing
  auto result = xTaskCreatePinnedToCore(
      bsec_task_, "bsec3", BSEC_TASK_STACK_SIZE, this,
      BSEC_TASK_PRIORITY, &this->task_handle_, 1);
  if (result != pdPASS) {
    this->mark_failed();
    ESP_LOGE(TAG, "Failed to create BSEC task");
    return;
  }

  ESP_LOGI(TAG, "BSEC3 task started");
}

void BME68xBSEC3Component::init_bsec_() {
  // Large buffers on the heap (BSEC work buffer + config blob are several KB)
  std::vector<uint8_t> work_buffer(BSEC_MAX_WORKBUFFER_SIZE);

  // Initialize BSEC library instance
  bsec_library_return_t bsec_status = bsec_init(this->bsec_instance_);
  if (bsec_status != BSEC_OK) {
    this->mark_failed();
    ESP_LOGE(TAG, "BSEC init failed: %d", bsec_status);
    return;
  }

  // Log version
  bsec_version_t version;
  bsec_get_version(this->bsec_instance_, &version);
  ESP_LOGI(TAG, "BSEC version: %d.%d.%d.%d", version.major, version.minor,
           version.major_bugfix, version.minor_bugfix);

  // Apply config blob
  if (this->bsec3_config_data_ != nullptr && this->bsec3_config_length_ > 0) {
    ESP_LOGI(TAG, "BSEC config blob: %u bytes (max property blob %u, work buffer %u)",
             (unsigned) this->bsec3_config_length_, (unsigned) BSEC_MAX_PROPERTY_BLOB_SIZE,
             (unsigned) BSEC_MAX_WORKBUFFER_SIZE);
    size_t cfg_size = std::max<size_t>(this->bsec3_config_length_, BSEC_MAX_PROPERTY_BLOB_SIZE);
    std::vector<uint8_t> config_buf(cfg_size);
    memcpy(config_buf.data(), this->bsec3_config_data_, this->bsec3_config_length_);

    bsec_status = bsec_set_configuration(
        this->bsec_instance_, config_buf.data(), this->bsec3_config_length_,
        work_buffer.data(), work_buffer.size());
    if (bsec_status != BSEC_OK) {
      ESP_LOGW(TAG, "BSEC set_configuration failed: %d", bsec_status);
    } else {
      ESP_LOGI(TAG, "BSEC config applied (%u bytes)", (unsigned) this->bsec3_config_length_);
    }
  }

  // Load saved state from NVS
  this->load_state_();

  // Subscribe to BSEC outputs
  this->subscribe_outputs_();
}

void BME68xBSEC3Component::subscribe_outputs_() {
  float sample_rate;
  switch (this->sample_rate_) {
    case SAMPLE_RATE_LP:
      sample_rate = BSEC_SAMPLE_RATE_LP;
      break;
    case SAMPLE_RATE_ULP:
      sample_rate = BSEC_SAMPLE_RATE_ULP;
      break;
    default:
      sample_rate = BSEC_SAMPLE_RATE_LP;
      break;
  }

  bsec_sensor_configuration_t requested[14];
  uint8_t n_requested = 0;

  requested[n_requested++] = {sample_rate, BSEC_OUTPUT_RAW_PRESSURE};
  requested[n_requested++] = {sample_rate, BSEC_OUTPUT_RAW_TEMPERATURE};
  requested[n_requested++] = {sample_rate, BSEC_OUTPUT_RAW_HUMIDITY};
  requested[n_requested++] = {sample_rate, BSEC_OUTPUT_RAW_GAS};
  requested[n_requested++] = {sample_rate, BSEC_OUTPUT_IAQ};
  requested[n_requested++] = {sample_rate, BSEC_OUTPUT_SENSOR_HEAT_COMPENSATED_TEMPERATURE};
  requested[n_requested++] = {sample_rate, BSEC_OUTPUT_SENSOR_HEAT_COMPENSATED_HUMIDITY};
  requested[n_requested++] = {sample_rate, BSEC_OUTPUT_STATIC_IAQ};
  requested[n_requested++] = {sample_rate, BSEC_OUTPUT_CO2_EQUIVALENT};
  // Note: BSEC_OUTPUT_BREATH_VOC_EQUIVALENT not subscribed — BSEC3 derives
  // breath VOC from CO2 equivalent internally, no separate subscription needed.
  // Adding it causes bsec_update_subscription() to fail.
  requested[n_requested++] = {sample_rate, BSEC_OUTPUT_STABILIZATION_STATUS};
  requested[n_requested++] = {sample_rate, BSEC_OUTPUT_RUN_IN_STATUS};
  requested[n_requested++] = {sample_rate, BSEC_OUTPUT_GAS_PERCENTAGE};

  // TVOC equivalent only available in LP mode
  if (this->sample_rate_ == SAMPLE_RATE_LP) {
    requested[n_requested++] = {sample_rate, BSEC_OUTPUT_TVOC_EQUIVALENT};
  }

  bsec_sensor_configuration_t required[BSEC_MAX_PHYSICAL_SENSOR];
  uint8_t n_required = BSEC_MAX_PHYSICAL_SENSOR;

  bsec_library_return_t status = bsec_update_subscription(
      this->bsec_instance_, requested, n_requested, required, &n_required);
  if (status != BSEC_OK) {
    ESP_LOGE(TAG, "BSEC update_subscription failed: %d", status);
  } else {
    ESP_LOGI(TAG, "BSEC subscribed to %d outputs", n_requested);
  }
}

// --- FreeRTOS Task ---

void BME68xBSEC3Component::bsec_task_(void *param) {
  auto *self = static_cast<BME68xBSEC3Component *>(param);

  // Wait for system stabilization
  vTaskDelay(pdMS_TO_TICKS(BSEC_TASK_STARTUP_DELAY_MS));
  ESP_LOGI(TAG, "BSEC task running");

  self->bsec_task_main_();
}

void BME68xBSEC3Component::bsec_task_main_() {
  bsec_bme_settings_t sensor_settings{};
  sensor_settings.next_call = 0;

  struct bme69x_data sensor_data[3];
  uint8_t n_fields;

  while (true) {
    int64_t time_ns = (int64_t) esp_timer_get_time() * INT64_C(1000);

    if (time_ns >= sensor_settings.next_call) {
      // All BSEC library calls of one cycle run under the BSEC mutex, so the
      // shutdown hook can never interrupt the algorithm in the middle of a step.
      xSemaphoreTake(this->bsec_mutex_, portMAX_DELAY);

      // Get BSEC sensor control settings
      bsec_library_return_t status = bsec_sensor_control(
          this->bsec_instance_, time_ns, &sensor_settings);
      if (status != BSEC_OK) {
        xSemaphoreGive(this->bsec_mutex_);
        ESP_LOGW(TAG, "bsec_sensor_control error: %d", status);
        vTaskDelay(pdMS_TO_TICKS(1000));
        continue;
      }

      // Configure sensor based on BSEC requirements
      switch (sensor_settings.op_mode) {
        case BME69X_FORCED_MODE: {
          bme69x_set_op_mode(BME69X_SLEEP_MODE, &this->bme69x_dev_);
          bme69x_get_conf(&this->bme69x_conf_, &this->bme69x_dev_);
          this->bme69x_conf_.os_hum = sensor_settings.humidity_oversampling;
          this->bme69x_conf_.os_temp = sensor_settings.temperature_oversampling;
          this->bme69x_conf_.os_pres = sensor_settings.pressure_oversampling;
          bme69x_set_conf(&this->bme69x_conf_, &this->bme69x_dev_);

          this->bme69x_heatr_conf_.enable = BME69X_ENABLE;
          this->bme69x_heatr_conf_.heatr_temp = sensor_settings.heater_temperature;
          this->bme69x_heatr_conf_.heatr_dur = sensor_settings.heater_duration;
          bme69x_set_heatr_conf(BME69X_FORCED_MODE, &this->bme69x_heatr_conf_, &this->bme69x_dev_);

          bme69x_set_op_mode(BME69X_FORCED_MODE, &this->bme69x_dev_);
          this->last_op_mode_ = BME69X_FORCED_MODE;
          this->current_op_mode_ = BME69X_FORCED_MODE;
          break;
        }
        case BME69X_PARALLEL_MODE: {
          if (this->current_op_mode_ != BME69X_PARALLEL_MODE) {
            bme69x_get_conf(&this->bme69x_conf_, &this->bme69x_dev_);
            this->bme69x_conf_.os_hum = sensor_settings.humidity_oversampling;
            this->bme69x_conf_.os_temp = sensor_settings.temperature_oversampling;
            this->bme69x_conf_.os_pres = sensor_settings.pressure_oversampling;
            bme69x_set_conf(&this->bme69x_conf_, &this->bme69x_dev_);

            uint16_t shared_dur = 140 - (bme69x_get_meas_dur(
                BME69X_PARALLEL_MODE, &this->bme69x_conf_, &this->bme69x_dev_) / 1000);
            this->bme69x_heatr_conf_.enable = BME69X_ENABLE;
            this->bme69x_heatr_conf_.heatr_temp_prof = sensor_settings.heater_temperature_profile;
            this->bme69x_heatr_conf_.heatr_dur_prof = sensor_settings.heater_duration_profile;
            this->bme69x_heatr_conf_.shared_heatr_dur = shared_dur;
            this->bme69x_heatr_conf_.profile_len = sensor_settings.heater_profile_len;
            bme69x_set_heatr_conf(BME69X_PARALLEL_MODE, &this->bme69x_heatr_conf_, &this->bme69x_dev_);

            bme69x_set_op_mode(BME69X_PARALLEL_MODE, &this->bme69x_dev_);
            this->last_op_mode_ = BME69X_PARALLEL_MODE;
            this->current_op_mode_ = BME69X_PARALLEL_MODE;
          }
          break;
        }
        case BME69X_SLEEP_MODE: {
          if (this->current_op_mode_ != BME69X_SLEEP_MODE) {
            bme69x_set_op_mode(BME69X_SLEEP_MODE, &this->bme69x_dev_);
            this->current_op_mode_ = BME69X_SLEEP_MODE;
          }
          break;
        }
        default:
          break;
      }

      // Read sensor data and process through BSEC
      if (sensor_settings.trigger_measurement && sensor_settings.op_mode != BME69X_SLEEP_MODE) {
        n_fields = 0;
        bme69x_get_data(
            this->last_op_mode_, &sensor_data[0], &n_fields, &this->bme69x_dev_);

        for (uint8_t i = 0; i < n_fields; i++) {
          if (sensor_data[i].status & BME69X_GASM_VALID_MSK) {
            this->process_sensor_data_(time_ns, &sensor_data[i], sensor_settings.process_data);
          }
        }
      }

      // Periodic state save - only with a fully calibrated state (accuracy 3).
      // If the interval is due while BSEC is still calibrating, the save is
      // retried every cycle until accuracy reaches 3 again.
      uint32_t now_ms = (uint32_t)(esp_timer_get_time() / 1000);
      if (now_ms - this->last_state_save_ms_ >= this->state_save_interval_ms_ && this->last_accuracy_ >= 3) {
        if (this->save_state_()) {
          this->last_state_save_ms_ = now_ms;
        }
      }

      xSemaphoreGive(this->bsec_mutex_);
    }

    // Sleep until next BSEC call
    int64_t now_us = esp_timer_get_time();
    int64_t next_us = sensor_settings.next_call / 1000;
    int64_t sleep_ms = (next_us - now_us) / 1000;
    if (sleep_ms > 10) {
      vTaskDelay(pdMS_TO_TICKS(sleep_ms));
    } else {
      vTaskDelay(pdMS_TO_TICKS(10));
    }
  }
}

void BME68xBSEC3Component::process_sensor_data_(int64_t time_ns, struct bme69x_data *data, uint32_t process_data) {
  bsec_input_t inputs[BSEC_MAX_PHYSICAL_SENSOR]{};
  uint8_t n_inputs = 0;

  if (process_data & BSEC_PROCESS_TEMPERATURE) {
    inputs[n_inputs].sensor_id = BSEC_INPUT_TEMPERATURE;
    inputs[n_inputs].signal = data->temperature;
    inputs[n_inputs].time_stamp = time_ns;
    n_inputs++;
  }
  if (process_data & BSEC_PROCESS_HUMIDITY) {
    inputs[n_inputs].sensor_id = BSEC_INPUT_HUMIDITY;
    inputs[n_inputs].signal = data->humidity;
    inputs[n_inputs].time_stamp = time_ns;
    n_inputs++;
  }
  if (process_data & BSEC_PROCESS_PRESSURE) {
    inputs[n_inputs].sensor_id = BSEC_INPUT_PRESSURE;
    inputs[n_inputs].signal = data->pressure;
    inputs[n_inputs].time_stamp = time_ns;
    n_inputs++;
  }
  if ((process_data & BSEC_PROCESS_GAS) && (data->status & BME69X_GASM_VALID_MSK)) {
    inputs[n_inputs].sensor_id = BSEC_INPUT_GASRESISTOR;
    inputs[n_inputs].signal = data->gas_resistance;
    inputs[n_inputs].time_stamp = time_ns;
    n_inputs++;
  }
  if ((process_data & BSEC_PROCESS_PROFILE_PART) && (data->status & BME69X_GASM_VALID_MSK)) {
    inputs[n_inputs].sensor_id = BSEC_INPUT_PROFILE_PART;
    inputs[n_inputs].signal = (this->current_op_mode_ == BME69X_FORCED_MODE) ? 0 : data->gas_index;
    inputs[n_inputs].time_stamp = time_ns;
    n_inputs++;
  }

  // Heat source compensation
  inputs[n_inputs].sensor_id = BSEC_INPUT_HEATSOURCE;
  inputs[n_inputs].signal = this->temperature_offset_;
  inputs[n_inputs].time_stamp = time_ns;
  n_inputs++;

  // Baseline tracker for LP mode
  if (this->sample_rate_ == SAMPLE_RATE_LP) {
    inputs[n_inputs].sensor_id = BSEC_INPUT_DISABLE_BASELINE_TRACKER;
    inputs[n_inputs].signal = 0;  // Baseline tracking enabled
    inputs[n_inputs].time_stamp = time_ns;
    n_inputs++;
  }

  if (n_inputs == 0) return;

  // Run BSEC algorithm
  bsec_output_t outputs[BSEC_NUMBER_OUTPUTS];
  uint8_t n_outputs = BSEC_NUMBER_OUTPUTS;

  bsec_library_return_t status = bsec_do_steps(
      this->bsec_instance_, inputs, n_inputs, outputs, &n_outputs);
  if (status != BSEC_OK) {
    ESP_LOGW(TAG, "bsec_do_steps error: %d", status);
    return;
  }

  if (n_outputs == 0) return;

  // Parse outputs and update shared struct
  SensorData new_data{};
  for (uint8_t i = 0; i < n_outputs; i++) {
    switch (outputs[i].sensor_id) {
      case BSEC_OUTPUT_RAW_TEMPERATURE:
        new_data.temperature = outputs[i].signal;
        break;
      case BSEC_OUTPUT_RAW_HUMIDITY:
        new_data.humidity = outputs[i].signal;
        break;
      case BSEC_OUTPUT_RAW_PRESSURE:
        new_data.pressure = outputs[i].signal / 100.0f;  // Pa to hPa
        break;
      case BSEC_OUTPUT_RAW_GAS:
        new_data.gas_resistance = outputs[i].signal;
        break;
      case BSEC_OUTPUT_IAQ:
        new_data.iaq = outputs[i].signal;
        new_data.iaq_accuracy = outputs[i].accuracy;
        this->last_accuracy_ = outputs[i].accuracy;
        break;
      case BSEC_OUTPUT_STATIC_IAQ:
        new_data.static_iaq = outputs[i].signal;
        break;
      case BSEC_OUTPUT_CO2_EQUIVALENT:
        new_data.co2_equivalent = outputs[i].signal;
        break;
      case BSEC_OUTPUT_BREATH_VOC_EQUIVALENT:
        new_data.breath_voc_equivalent = outputs[i].signal;
        break;
      case BSEC_OUTPUT_SENSOR_HEAT_COMPENSATED_TEMPERATURE:
        new_data.compensated_temperature = outputs[i].signal;
        break;
      case BSEC_OUTPUT_SENSOR_HEAT_COMPENSATED_HUMIDITY:
        new_data.compensated_humidity = outputs[i].signal;
        break;
      case BSEC_OUTPUT_GAS_PERCENTAGE:
        new_data.gas_percentage = outputs[i].signal;
        break;
      case BSEC_OUTPUT_TVOC_EQUIVALENT:
        new_data.tvoc_equivalent = outputs[i].signal;
        break;
      default:
        break;
    }
  }
  new_data.valid = true;

  // Copy to shared struct under mutex
  if (this->data_mutex_ != nullptr) {
    xSemaphoreTake(this->data_mutex_, portMAX_DELAY);
  }
  this->sensor_data_ = new_data;
  this->data_available_ = true;
  if (this->data_mutex_ != nullptr) {
    xSemaphoreGive(this->data_mutex_);
  }
}

// --- Main Loop (publishes sensor values) ---

void BME68xBSEC3Component::loop() {
}

void BME68xBSEC3Component::update() {
  if (!this->data_available_) {
    return;
  }

  // Copy data under mutex protection, then release immediately
  SensorData data;
  if (this->data_mutex_ != nullptr) {
    xSemaphoreTake(this->data_mutex_, portMAX_DELAY);
  }
  data = this->sensor_data_;
  if (this->data_mutex_ != nullptr) {
    xSemaphoreGive(this->data_mutex_);
  }

  // Accuracy 0 = BSEC is stabilizing (first ~5 min after boot, also with a
  // restored state): the IAQ-derived outputs ramp up from 0 and mean nothing yet.
  const bool stabilizing = data.iaq_accuracy == 0;

  ESP_LOGI(TAG, "T=%.1f H=%.1f P=%.1f IAQ=%.0f acc=%d%s",
           data.compensated_temperature, data.compensated_humidity, data.pressure,
           data.iaq, data.iaq_accuracy, stabilizing ? " (stabilizing, IAQ hidden)" : "");

#ifdef USE_SENSOR
  if (this->temperature_sensor_ != nullptr)
    this->temperature_sensor_->publish_state(data.compensated_temperature);
  if (this->humidity_sensor_ != nullptr)
    this->humidity_sensor_->publish_state(data.compensated_humidity);
  if (this->pressure_sensor_ != nullptr)
    this->pressure_sensor_->publish_state(data.pressure);
  if (this->gas_resistance_sensor_ != nullptr)
    this->gas_resistance_sensor_->publish_state(data.gas_resistance);
  if (this->iaq_sensor_ != nullptr)
    this->iaq_sensor_->publish_state(stabilizing ? NAN : data.iaq);
  if (this->iaq_accuracy_sensor_ != nullptr)
    this->iaq_accuracy_sensor_->publish_state((float) data.iaq_accuracy);
  if (this->static_iaq_sensor_ != nullptr)
    this->static_iaq_sensor_->publish_state(stabilizing ? NAN : data.static_iaq);
  if (this->co2_equivalent_sensor_ != nullptr)
    this->co2_equivalent_sensor_->publish_state(stabilizing ? NAN : data.co2_equivalent);
  if (this->breath_voc_equivalent_sensor_ != nullptr)
    this->breath_voc_equivalent_sensor_->publish_state(stabilizing ? NAN : data.breath_voc_equivalent);
  if (this->gas_percentage_sensor_ != nullptr)
    this->gas_percentage_sensor_->publish_state(stabilizing ? NAN : data.gas_percentage);
  if (this->tvoc_equivalent_sensor_ != nullptr)
    this->tvoc_equivalent_sensor_->publish_state(stabilizing ? NAN : data.tvoc_equivalent);
  if (this->compensated_temperature_sensor_ != nullptr)
    this->compensated_temperature_sensor_->publish_state(data.temperature);
  if (this->compensated_humidity_sensor_ != nullptr)
    this->compensated_humidity_sensor_->publish_state(data.humidity);
#endif
#ifdef USE_TEXT_SENSOR
  if (this->iaq_accuracy_text_sensor_ != nullptr) {
    const char *accuracy_text;
    switch (data.iaq_accuracy) {
      case 0: accuracy_text = "Stabilizing"; break;
      case 1: accuracy_text = "Uncertain"; break;
      case 2: accuracy_text = "Calibrating"; break;
      case 3: accuracy_text = "Calibrated"; break;
      default: accuracy_text = "Unknown"; break;
    }
    this->iaq_accuracy_text_sensor_->publish_state(accuracy_text);
  }
  if (this->iaq_description_text_sensor_ != nullptr && !stabilizing) {
    const char *desc;
    if (data.iaq <= 50)
      desc = "Excellent";
    else if (data.iaq <= 100)
      desc = "Good";
    else if (data.iaq <= 150)
      desc = "Lightly Polluted";
    else if (data.iaq <= 200)
      desc = "Moderately Polluted";
    else if (data.iaq <= 250)
      desc = "Heavily Polluted";
    else if (data.iaq <= 350)
      desc = "Severely Polluted";
    else
      desc = "Extremely Polluted";
    this->iaq_description_text_sensor_->publish_state(desc);
  }
#endif
}

// --- Safe shutdown (OTA, restart button, API/WiFi reboot timeout) ---

void BME68xBSEC3Component::on_safe_shutdown() {
  if (this->bsec_instance_ == nullptr || this->bsec_mutex_ == nullptr || this->task_handle_ == nullptr) {
    return;
  }
  // Wait until the BSEC task has finished its current cycle, then stop it so
  // the instance cannot change while the state is read.
  if (xSemaphoreTake(this->bsec_mutex_, pdMS_TO_TICKS(BSEC_SHUTDOWN_LOCK_TIMEOUT_MS)) != pdTRUE) {
    ESP_LOGW(TAG, "Shutdown: BSEC busy, state not saved");
    return;
  }
  vTaskSuspend(this->task_handle_);
  if (this->last_accuracy_ >= 3) {
    ESP_LOGI(TAG, "Shutdown: saving calibrated BSEC state");
    this->save_state_();
  } else {
    ESP_LOGI(TAG, "Shutdown: IAQ accuracy %d < 3, keeping the last calibrated state",
             (int) this->last_accuracy_);
  }
  xSemaphoreGive(this->bsec_mutex_);
}

// --- State Persistence ---

typedef uint8_t BsecStateBlob[BSEC_MAX_STATE_BLOB_SIZE];

bool BME68xBSEC3Component::save_state_() {
  std::vector<uint8_t> state_buffer(BSEC_MAX_STATE_BLOB_SIZE);
  std::vector<uint8_t> work_buffer(BSEC_MAX_WORKBUFFER_SIZE);
  uint32_t state_length = 0;

  bsec_library_return_t status = bsec_get_state(
      this->bsec_instance_, 0, state_buffer.data(), state_buffer.size(),
      work_buffer.data(), work_buffer.size(), &state_length);
  if (status != BSEC_OK) {
    ESP_LOGW(TAG, "Failed to get BSEC state: %d", status);
    return false;
  }
  if (state_length == 0 || state_length > BSEC_MAX_STATE_BLOB_SIZE) {
    ESP_LOGW(TAG, "BSEC state has invalid size: %u", (unsigned) state_length);
    return false;
  }

  // Save to NVS via ESPHome preferences
  auto pref = global_preferences->make_preference<BsecStateBlob>(BSEC_STATE_HASH);
  std::vector<uint8_t> save_buf(BSEC_MAX_STATE_BLOB_SIZE, 0);
  memcpy(save_buf.data(), state_buffer.data(), state_length);
  if (!pref.save(reinterpret_cast<BsecStateBlob *>(save_buf.data()))) {
    ESP_LOGW(TAG, "Failed to write BSEC state to flash");
    return false;
  }
  global_preferences->sync();

  ESP_LOGI(TAG, "BSEC state saved (%u bytes, accuracy %d)", (unsigned) state_length,
           (int) this->last_accuracy_);
  return true;
}

void BME68xBSEC3Component::load_state_() {
  auto pref = global_preferences->make_preference<BsecStateBlob>(BSEC_STATE_HASH);
  std::vector<uint8_t> state_buffer(BSEC_MAX_STATE_BLOB_SIZE, 0);

  if (!pref.load(reinterpret_cast<BsecStateBlob *>(state_buffer.data()))) {
    ESP_LOGI(TAG, "No saved BSEC state found");
    return;
  }

  std::vector<uint8_t> work_buffer(BSEC_MAX_WORKBUFFER_SIZE);
  bsec_library_return_t status = bsec_set_state(
      this->bsec_instance_, state_buffer.data(), BSEC_MAX_STATE_BLOB_SIZE,
      work_buffer.data(), work_buffer.size());
  if (status != BSEC_OK) {
    ESP_LOGW(TAG, "Failed to restore BSEC state: %d", status);
  } else {
    ESP_LOGI(TAG, "BSEC state restored");
  }
}

// --- Dump Config ---

void BME68xBSEC3Component::dump_config() {
  ESP_LOGCONFIG(TAG, "BME68x BSEC3:");
  ESP_LOGCONFIG(TAG, "  Chip ID: 0x%02X", this->bme69x_dev_.chip_id);
  ESP_LOGCONFIG(TAG, "  Sample Rate: %s", this->sample_rate_ == SAMPLE_RATE_LP ? "LP (3s)" : "ULP (300s)");
  ESP_LOGCONFIG(TAG, "  Supply Voltage: %s", this->supply_voltage_ == SUPPLY_VOLTAGE_3V3 ? "3.3V" : "1.8V");
  ESP_LOGCONFIG(TAG, "  Operating Age: %s", this->operating_age_ == OPERATING_AGE_28D ? "28 days" : "4 days");
  ESP_LOGCONFIG(TAG, "  Temperature Offset: %.2f", this->temperature_offset_);
  ESP_LOGCONFIG(TAG, "  State Save Interval: %us (only at IAQ accuracy 3, plus on safe shutdown)",
                (unsigned) (this->state_save_interval_ms_ / 1000));

  if (this->bsec_instance_ != nullptr) {
    bsec_version_t version;
    if (bsec_get_version(this->bsec_instance_, &version) == BSEC_OK) {
      ESP_LOGCONFIG(TAG, "  BSEC Version: %d.%d.%d.%d", version.major, version.minor,
                    version.major_bugfix, version.minor_bugfix);
    }
  }

  LOG_I2C_DEVICE(this);
  LOG_UPDATE_INTERVAL(this);
}

}  // namespace bme68x_bsec3
}  // namespace esphome
