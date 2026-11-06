/*
  Lifora ESP32 alert-output adapter example.

  IMPORTANT: This is an additive adapter example, not a replacement for your
  current sensor loop or existing ML fall-detection code. Merge the pin setup,
  network task and output helpers into your existing firmware. Do not replace
  sensor sampling, preprocessing, or inference. Requires ArduinoJson 6.x.
*/
#include <WiFi.h>
#include <HTTPClient.h>
#include <ArduinoJson.h>

const char* WIFI_SSID = "SET_WIFI_SSID";
const char* WIFI_PASSWORD = "SET_WIFI_PASSWORD";
const char* API_BASE = "http://192.168.1.10:8000"; // host LAN IP, not 127.0.0.1
const char* DEVICE_TOKEN = "SET_DEVICE_INGEST_TOKEN";
const char* PATIENT_ID = "demo_patient";
const char* DEVICE_ID = "lifora-esp32-01";

const int BUZZER_PIN = 25;
const int LED_RED_PIN = 26;
const int LED_AMBER_PIN = 27;
const int LED_GREEN_PIN = 14;
const int PERSON_CHECKIN_BUTTON_PIN = 33; // button from pin to GND, INPUT_PULLUP

enum LedState : uint8_t { LED_AMBER = 0, LED_GREEN = 1, LED_RED = 2 };
enum AlarmState : uint8_t { ALARM_SILENT = 0, ALARM_WARNING = 1, ALARM_CRITICAL = 2 };
volatile uint8_t gLedState = LED_AMBER;
volatile uint8_t gAlarmState = ALARM_SILENT;
volatile bool gAlarmActive = false;
volatile bool gPersonResponsePending = false;

unsigned long gLastBeepMs = 0;
unsigned long gBeepOnUntilMs = 0;
bool gButtonLastRead = HIGH;
bool gButtonStableState = HIGH;
unsigned long gLastButtonChangeMs = 0;

void liforaSetLeds(uint8_t color) {
  digitalWrite(LED_RED_PIN, color == LED_RED ? HIGH : LOW);
  digitalWrite(LED_AMBER_PIN, color == LED_AMBER ? HIGH : LOW);
  digitalWrite(LED_GREEN_PIN, color == LED_GREEN ? HIGH : LOW);
}

// Runs on a separate FreeRTOS task so network delays do not block the sensor/ML loop.
void liforaNetworkTask(void* parameter) {
  for (;;) {
    if (WiFi.status() != WL_CONNECTED) {
      // Do not clear an already active local critical alarm because Wi-Fi is down.
      if (gLedState == LED_RED || (gAlarmState == ALARM_CRITICAL && gAlarmActive)) gLedState = LED_RED;
      else gLedState = LED_AMBER;
      vTaskDelay(pdMS_TO_TICKS(1000));
      continue;
    }

    HTTPClient http;
    String url = String(API_BASE) + "/api/v1/devices/alert-state?patient_id=" + PATIENT_ID;
    http.begin(url);
    http.setTimeout(2500);
    http.addHeader("X-Device-Token", DEVICE_TOKEN);
    const int code = http.GET();
    if (code == 200) {
      const String body = http.getString();
      StaticJsonDocument<768> doc;
      const DeserializationError error = deserializeJson(doc, body);
      if (!error) {
        const char* led = doc["led_color"] | "amber";
        const char* pattern = doc["alarm_pattern"] | "silent";
        gLedState = String(led) == "red" ? LED_RED : (String(led) == "green" ? LED_GREEN : LED_AMBER);
        gAlarmState = String(pattern) == "critical" ? ALARM_CRITICAL : (String(pattern) == "warning" ? ALARM_WARNING : ALARM_SILENT);
        gAlarmActive = doc["alarm_active"] | false;
      } else {
        if (!(gLedState == LED_RED || (gAlarmState == ALARM_CRITICAL && gAlarmActive))) gLedState = LED_AMBER;
      }
    } else {
      // Keep the last alarm state if the API/network fails; never show green on failure.
      if (!(gLedState == LED_RED || (gAlarmState == ALARM_CRITICAL && gAlarmActive))) gLedState = LED_AMBER;
    }
    http.end();

    // Send a physical check-in button response from this network task.
    if (gPersonResponsePending) {
      HTTPClient post;
      post.begin(String(API_BASE) + "/api/v1/devices/ingest");
      post.setTimeout(2500);
      post.addHeader("Content-Type", "application/json");
      post.addHeader("X-Device-Token", DEVICE_TOKEN);
      StaticJsonDocument<192> payload;
      payload["patient_id"] = PATIENT_ID;
      payload["device_id"] = DEVICE_ID;
      payload["person_response"] = true;
      String body;
      serializeJson(payload, body);
      const int postCode = post.POST(body);
      if ((postCode >= 200 && postCode < 300) || postCode == 409) gPersonResponsePending = false;
      post.end();
    }

    vTaskDelay(pdMS_TO_TICKS(3000));
  }
}

// Call this often from the existing loop; it does not perform network I/O.
void liforaUpdateOutputs() {
  liforaSetLeds(gLedState);
  const unsigned long now = millis();
  const uint8_t alarm = gAlarmState;
  if (!gAlarmActive || alarm == ALARM_SILENT) {
    noTone(BUZZER_PIN);
    return;
  }
  const unsigned long interval = alarm == ALARM_CRITICAL ? 1000 : 1700;
  const unsigned long pulse = alarm == ALARM_CRITICAL ? 450 : 180;
  if (now - gLastBeepMs >= interval) {
    gLastBeepMs = now;
    tone(BUZZER_PIN, alarm == ALARM_CRITICAL ? 2400 : 1500);
    gBeepOnUntilMs = now + pulse;
  }
  if (gBeepOnUntilMs && now >= gBeepOnUntilMs) {
    noTone(BUZZER_PIN);
    gBeepOnUntilMs = 0;
  }
}

// Call this often from the existing loop; a press queues a request for the network task.
void liforaPollPersonCheckinButton() {
  const unsigned long now = millis();
  const bool buttonState = digitalRead(PERSON_CHECKIN_BUTTON_PIN);
  if (buttonState != gButtonLastRead) {
    gLastButtonChangeMs = now;
    gButtonLastRead = buttonState;
  }
  if (now - gLastButtonChangeMs > 60 && buttonState != gButtonStableState) {
    gButtonStableState = buttonState;
    if (gButtonStableState == LOW) gPersonResponsePending = true; // one event per press
  }
}

void liforaAlertAdapterSetup() {
  pinMode(BUZZER_PIN, OUTPUT);
  pinMode(LED_RED_PIN, OUTPUT);
  pinMode(LED_AMBER_PIN, OUTPUT);
  pinMode(LED_GREEN_PIN, OUTPUT);
  pinMode(PERSON_CHECKIN_BUTTON_PIN, INPUT_PULLUP);
  liforaSetLeds(LED_AMBER); // unknown until a successful API response
  noTone(BUZZER_PIN);
  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  xTaskCreatePinnedToCore(liforaNetworkTask, "lifora-alert-network", 8192, nullptr, 1, nullptr, 0);
}

// Standalone demo only. In the actual project, call liforaAlertAdapterSetup()
// from the existing setup(), and call both poll/update helpers from the existing
// loop() without replacing any existing sensor collection or ML inference code.
void setup() {
  liforaAlertAdapterSetup();
}

void loop() {
  liforaPollPersonCheckinButton();
  liforaUpdateOutputs();
  delay(2); // cooperative yield; no long blocking sensor delays
}
