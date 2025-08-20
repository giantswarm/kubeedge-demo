package driver

import (
	"encoding/json"
	"encoding/xml"
	"errors"
	"fmt"
	"strconv"
	"strings"
	"sync"
	"time"

	mqtt "github.com/eclipse/paho.mqtt.golang"
	"gopkg.in/yaml.v3"
	"k8s.io/klog/v2"

	"github.com/kubeedge/mapper-framework/pkg/common"
)

func NewClient(protocol ProtocolConfig) (*CustomizedClient, error) {
	client := &CustomizedClient{
		ProtocolConfig:   protocol,
		deviceMutex:      sync.Mutex{},
		TempMessage:      "",
		DeviceConfigData: nil,
	}
	return client, nil
}

func (c *CustomizedClient) InitDevice() error {
	configData := &c.ProtocolConfig.ConfigData
	_, operationInfo, serializedFormat, err := configData.SplitTopic()
	if operationInfo != DEVICEINfO {
		return errors.New("this is not a device config")
	}
	if err != nil {
		return err
	}
	c.TempMessage = configData.Message

	// Parse the message to initialize DeviceConfigData
	parsedMessage, err := configData.ParseMessage(serializedFormat)
	if err != nil {
		return fmt.Errorf("failed to parse device config message: %v", err)
	}

	// Initialize the device data as a map to support dynamic properties
	deviceData := make(MQTTDeviceData)

	// Populate the map from the parsed message - any property is now supported
	for key, value := range parsedMessage {
		deviceData[key] = value
	}

	// Set default status if not provided (keep this for backward compatibility)
	if _, exists := deviceData["status"]; !exists {
		deviceData["status"] = "online"
	}

	c.DeviceConfigData = deviceData

	// Initialize MQTT client for receiving data updates
	err = c.initMQTTSubscription()
	if err != nil {

		klog.Errorf("Failed to initialize MQTT subscription: %v", err)
		// Don't return error - device can still work without MQTT updates
	}

	return nil
}

func (c *CustomizedClient) GetDeviceData(visitor *VisitorConfig) (interface{}, error) {
	// For twin reporting, we should return current device data regardless of protocol operation
	c.dataMutex.RLock()
	defer c.dataMutex.RUnlock()

	if c.DeviceConfigData == nil {
		return nil, errors.New("device config data is not initialized")
	}

	klog.V(3).Infof("GetDeviceData() returning data (type: %T)", c.DeviceConfigData)
	// We need to return the entire device data because the TwinData context
	// will extract the correct field based on the property name.
	return c.DeviceConfigData, nil
}

func (c *CustomizedClient) SetDeviceData(visitor *VisitorConfig) error {
	// Check the visitor's operation type directly instead of using protocol config
	if visitor.VisitorConfigData.OperationInfo == DEVICEINfO {
		return errors.New("this is a device config, not to set device data")
	}
	visitor.ProcessOperation(c.DeviceConfigData)
	return nil
}

func (c *CustomizedClient) StopDevice() error {
	// Disconnect MQTT client if connected
	if c.mqttClient != nil && c.mqttClient.IsConnected() {
		c.mqttClient.Disconnect(250)
		klog.Info("Disconnected from MQTT broker")
	}

	// Update status in the device data map
	c.dataMutex.Lock()
	defer c.dataMutex.Unlock()

	if c.DeviceConfigData != nil {
		c.DeviceConfigData["status"] = common.DeviceStatusDisCONN
	}
	return nil
}

func (c *CustomizedClient) GetDeviceStates(visitor *VisitorConfig) (string, error) {
	c.dataMutex.RLock()
	defer c.dataMutex.RUnlock()

	if c.DeviceConfigData == nil {
		return common.DeviceStatusOK, nil
	}

	// Check for status field in the device data
	if status, exists := c.DeviceConfigData["status"]; exists {
		if statusStr, ok := status.(string); ok {
			return statusStr, nil
		}
	}

	return common.DeviceStatusOK, nil
}

// initMQTTSubscription initializes MQTT client and subscribes to device data topics
func (c *CustomizedClient) initMQTTSubscription() error {
	// Create MQTT client options
	opts := mqtt.NewClientOptions()
	opts.AddBroker(c.ProtocolConfig.ConfigData.BrokerURL)
	opts.SetClientID(c.ProtocolConfig.ConfigData.ClientID + "_sub")
	opts.SetUsername(c.ProtocolConfig.ConfigData.Username)
	opts.SetPassword(c.ProtocolConfig.ConfigData.Password)
	opts.SetAutoReconnect(true)
	opts.SetCleanSession(true)

	// Create and connect MQTT client
	c.mqttClient = mqtt.NewClient(opts)
	if token := c.mqttClient.Connect(); token.Wait() && token.Error() != nil {
		return fmt.Errorf("failed to connect to MQTT broker: %v", token.Error())
	}

	klog.Infof("Connected to MQTT broker: %s", c.ProtocolConfig.ConfigData.BrokerURL)

	// Subscribe to data update topics based on the device topic pattern
	// Extract device path from protocol topic: "sensor/beta1-device/deviceinfo/json" -> "sensor/beta1-device"
	topicParts := strings.Split(c.ProtocolConfig.ConfigData.Topic, "/")
	if len(topicParts) >= 2 {
		devicePath := strings.Join(topicParts[:len(topicParts)-2], "/")

		// Subscribe to topics where device data updates will be published
		dataTopics := []string{
			devicePath + "/update/json",
			devicePath + "/getsinglevalue/json",
			devicePath + "/setsinglevalue/json",
		}

		for _, topic := range dataTopics {
			if token := c.mqttClient.Subscribe(topic, 1, c.onMQTTMessage); token.Wait() && token.Error() != nil {
				klog.Errorf("Failed to subscribe to topic %s: %v", topic, token.Error())
			} else {
				klog.V(2).Infof("Subscribed to MQTT topic: %s", topic)
			}
		}
	}

	return nil
}

// onMQTTMessage handles incoming MQTT messages and updates device data
func (c *CustomizedClient) onMQTTMessage(client mqtt.Client, msg mqtt.Message) {
	klog.V(2).Infof("Received MQTT message on topic %s: %s", msg.Topic(), string(msg.Payload()))

	// Parse the incoming message
	var messageData map[string]interface{}
	if err := json.Unmarshal(msg.Payload(), &messageData); err != nil {
		klog.Errorf("Failed to parse MQTT message: %v", err)
		return
	}

	// Update device data in thread-safe manner
	c.dataMutex.Lock()
	defer c.dataMutex.Unlock()

	if c.DeviceConfigData == nil {
		c.DeviceConfigData = make(MQTTDeviceData)
	}

	// Update any properties present in the message
	for key, value := range messageData {
		// Convert different value types to appropriate format
		var convertedValue interface{}
		switch v := value.(type) {
		case string:
			convertedValue = v
		case float64:
			convertedValue = fmt.Sprintf("%.1f", v)
		case int:
			convertedValue = fmt.Sprintf("%d", v)
		case bool:
			convertedValue = fmt.Sprintf("%t", v)
		default:
			convertedValue = fmt.Sprintf("%v", v)
		}

		c.DeviceConfigData[key] = convertedValue
		klog.V(2).Infof("Updated device property %s to: %v", key, convertedValue)
	}
}

/* --------------------------------------------------------------------------------------- */
// The function NewConfigData is a constructor for ConfigData to initialize the structure.
// It returns the ConfigData instance and an error value to handle the validity of the passed parameters.
func NewConfigData(clientID, brokerURL, topic, message, username, password string, connectionTTL time.Duration) (*ConfigData, error) {
	if clientID == "" {
		return nil, errors.New("clientID cannot be empty")
	}
	if brokerURL == "" {
		return nil, errors.New("borkerURL cannot be empty")
	}
	if topic == "" {
		return nil, errors.New("topic cannot be empty")
	}
	if message == "" {
		return nil, errors.New("message cannot be empty")
	}
	if username == "" {
		username = "defaultUser"
	}
	if password == "" {
		password = "defaultPass"
	}
	if connectionTTL == 0 {
		connectionTTL = 30 * time.Second // default timeout of 30 seconds
	}

	return &ConfigData{
		ClientID:      clientID,
		BrokerURL:     brokerURL,
		Topic:         topic,
		Message:       message,
		Username:      username,
		Password:      password,
		ConnectionTTL: connectionTTL,
		LastMessage:   time.Now(), // set last message time to current time
	}, nil
}

// The function GetClientID returns the value of the ClientID field and error.
func (c *ConfigData) GetClientID() (string, error) {
	if c.ClientID == "" {
		return "", errors.New("clientID is empty")
	}
	return c.ClientID, nil
}

// The function GetTopic returns the value of the Topic field and error.
func (c *ConfigData) GetTopic() (string, error) {
	if c.Topic == "" {
		return "", errors.New("topic is empty")
	}
	return c.Topic, nil
}

// GetMessage returns the value of the Message field and error.
func (c *ConfigData) GetMessage() (string, error) {
	if c.Message == "" {
		return "", errors.New("message is empty")
	}
	return c.Message, nil
}

// OperationInfoType and SerializedFormatType mappings
var operationTypeMap = map[string]OperationInfoType{
	"update":         UPDATE,
	"deviceinfo":     DEVICEINfO,
	"setsinglevalue": SETSINGLEVALUE,
	"getsinglevalue": GETSINGLEVALUE,
}

var serializedFormatMap = map[string]SerializedFormatType{
	"json": JSON,
	"yaml": YAML,
	"xml":  XML,
}

// The function SplitTopic splits the Topic into three parts and returns each.
// OperationInfoType(fulltextmodify: 0, pathmodify: 1, valuemodify: 2)
// SerializedFormatType(json: 0, yaml: 1, xml: 2)
func (c *ConfigData) SplitTopic() (string, OperationInfoType, SerializedFormatType, error) {
	if c.Topic == "" {
		return "", 0, 0, errors.New("topic is empty")
	}

	parts := strings.Split(c.Topic, "/")

	if len(parts) < 3 {
		return "", 0, 0, errors.New("topic format is invalid, must have at least three parts")
	}

	deviceInfo := strings.Join(parts[:len(parts)-2], "/")

	// Get operation type from map
	operationType, exists := operationTypeMap[parts[len(parts)-2]]
	if !exists {
		return "", 0, 0, errors.New("invalid operation type")
	}

	// Get serialized format from map
	serializedFormat, exists := serializedFormatMap[parts[len(parts)-1]]
	if !exists {
		return "", 0, 0, errors.New("invalid serialized format")
	}

	return deviceInfo, operationType, serializedFormat, nil
}

// The function ParseMessage parses the Message field according to the incoming type.
// parseType(0: json, 1: yaml, 2: xml)
// The value interface{} represents the parsed structure.
func (c *ConfigData) ParseMessage(parseType SerializedFormatType) (map[string]interface{}, error) {
	if c.Message == "" {
		return nil, errors.New("message is empty")
	}

	switch parseType {
	case JSON: // json
		return c.jsonParse()

	case YAML: // yaml
		return c.yamlParse()

	case XML: // xml
		return c.xmlParse()

	default:
		return nil, errors.New("unsupported parse type")
	}
}

// The function parseJSON parses the Message field of the ConfigData (assumed to be a JSON string).
func (c *ConfigData) jsonParse() (map[string]interface{}, error) {
	if c.Message == "" {
		return nil, errors.New("message is empty")
	}

	var jsonMsg map[string]interface{}
	err := json.Unmarshal([]byte(c.Message), &jsonMsg)
	if err != nil {
		return nil, err
	}
	return jsonMsg, nil
}

// The function parseYAML parses the Message field of the ConfigData (assumed to be a YAML string).
func (c *ConfigData) yamlParse() (map[string]interface{}, error) {
	if c.Message == "" {
		return nil, errors.New("message is empty")
	}

	var yamlMsg map[string]interface{}
	err := yaml.Unmarshal([]byte(c.Message), &yamlMsg)
	if err != nil {
		return nil, err
	}
	return yamlMsg, nil
}

// The function xmlParse parses the Message field of the ConfigData (assumed to be a XML string).
func (c *ConfigData) xmlParse() (map[string]interface{}, error) {
	msg := c.Message
	if strings.HasPrefix(msg, "<?xml") {
		end := strings.Index(msg, "?>")
		if end != -1 {
			msg = msg[end+2:]
		}
	}

	var node Node
	err := xml.Unmarshal([]byte(msg), &node)
	if err != nil {
		return nil, err
	}

	xmlMsg := nodeToMap(node)
	var mp map[string]interface{}
	for _, value := range xmlMsg {
		if nestedMap, ok := value.(map[string]interface{}); ok {
			mp = nestedMap
			break
		}
	}
	return mp, err
}

// NewVisitorConfig creates a new instance of VisitorConfig using ConfigData pointer and the result of SplitTopic.
func (c *ConfigData) NewVisitorConfig() (*VisitorConfig, error) {
	// get ClientID
	clientID, err := c.GetClientID()
	if err != nil {
		return nil, err
	}

	// get DeviceInfo, OperationInfo and SerializedFormat
	deviceInfo, operationInfo, serializedFormat, err := c.SplitTopic()
	if err != nil {
		return nil, err
	}

	// get ParsedMessage
	parsedMessage, err := c.ParseMessage(serializedFormat)
	if err != nil {
		return nil, err
	}

	// create
	return &VisitorConfig{
		ProtocolName: "mqtt",
		VisitorConfigData: VisitorConfigData{
			DataType:         "DefaultDataType",
			ClientID:         clientID,
			DeviceInfo:       deviceInfo,
			OperationInfo:    operationInfo,
			SerializedFormat: serializedFormat,
			ParsedMessage:    parsedMessage,
		},
	}, nil
}

/* --------------------------------------------------------------------------------------- */
// The function ParseMessage parses the Message field according to the incoming type.
// parseType(0: json, 1: yaml, 2: xml)
// The value interface{} represents the parsed structure.
func (v *VisitorConfig) ProcessOperation(deviceConfigData interface{}) error {
	if v.VisitorConfigData.ParsedMessage == nil {
		return errors.New("visitor message is empty")
	}

	if deviceConfigData == nil {
		return errors.New("device message is empty")
	}

	switch v.VisitorConfigData.OperationInfo {
	case DEVICEINfO: // device config data
		v.updateFullConfig(deviceConfigData)
		return nil
	case UPDATE: // update the full text according the visitor config and the tag (json, yaml, xml)
		v.updateFullConfig(deviceConfigData)
		return nil
	case SETSINGLEVALUE: // update the single value according the visitor config and the tag (json, yaml, xml)
		v.updateFieldsByTag(deviceConfigData)
		return nil
	default:
		return errors.New("unsupported operation type")
	}
}

func (v *VisitorConfig) updateFullConfig(destDataConfig interface{}) error {
	// Handle MQTTDeviceData map
	if deviceMap, ok := destDataConfig.(MQTTDeviceData); ok {
		// Clear existing data and replace with new data
		for k := range deviceMap {
			delete(deviceMap, k)
		}
		for key, value := range v.VisitorConfigData.ParsedMessage {
			deviceMap[key] = value
		}
		return nil
	}

	// Handle pointer to MQTTDeviceData map
	if deviceMapPtr, ok := destDataConfig.(*MQTTDeviceData); ok && deviceMapPtr != nil {
		// Clear existing data and replace with new data
		for k := range *deviceMapPtr {
			delete(*deviceMapPtr, k)
		}
		for key, value := range v.VisitorConfigData.ParsedMessage {
			(*deviceMapPtr)[key] = value
		}
		return nil
	}

	return errors.New("destDataConfig must be MQTTDeviceData map")
}

func (v *VisitorConfig) updateFieldsByTag(destDataConfig interface{}) error {
	// Convert to map if it's MQTTDeviceData
	if deviceMap, ok := destDataConfig.(MQTTDeviceData); ok {
		// Update the map directly
		for key, value := range v.VisitorConfigData.ParsedMessage {
			deviceMap[key] = value
		}
		return nil
	}

	// If it's a pointer to map, dereference it
	if deviceMapPtr, ok := destDataConfig.(*MQTTDeviceData); ok && deviceMapPtr != nil {
		for key, value := range v.VisitorConfigData.ParsedMessage {
			(*deviceMapPtr)[key] = value
		}
		return nil
	}

	return errors.New("destDataConfig must be MQTTDeviceData map")
}

// Node structure
type Node struct {
	XMLName xml.Name
	Content string     `xml:",chardata"`
	Nodes   []Node     `xml:",any"`
	Attr    []xml.Attr `xml:"-"`
}

// convertValue attempts to convert string content to appropriate type.
func convertValue(content string) interface{} {
	if f, err := strconv.ParseFloat(content, 64); err == nil {
		return f
	} else if i, err := strconv.Atoi(content); err == nil {
		return i
	} else if b, err := strconv.ParseBool(content); err == nil {
		return b
	} else {
		return content
	}
}

// Convert XML attributes to map entries
func attrsToMap(attrs []xml.Attr) map[string]interface{} {
	attrMap := make(map[string]interface{})
	for _, attr := range attrs {
		attrMap[attr.Name.Local] = attr.Value
	}
	return attrMap
}

// The function nodeToMap recursively converts XML nodes to map[string]interface{}.
func nodeToMap(node Node) map[string]interface{} {
	xmlMsg := make(map[string]interface{})

	// Process attributes
	if len(node.Attr) > 0 {
		xmlMsg["attributes"] = attrsToMap(node.Attr)
	}

	// If the node has no children, it is a leaf node, apply type conversion.
	if len(node.Nodes) == 0 {
		xmlMsg[node.XMLName.Local] = convertValue(strings.TrimSpace(node.Content))
		return xmlMsg
	}

	// Process child nodes recursively.
	children := make(map[string]interface{})
	for _, child := range node.Nodes {
		childMap := nodeToMap(child)
		if existing, found := children[child.XMLName.Local]; found {
			switch v := existing.(type) {
			case []interface{}:
				children[child.XMLName.Local] = append(v, childMap[child.XMLName.Local])
			default:
				children[child.XMLName.Local] = []interface{}{v, childMap[child.XMLName.Local]}
			}
		} else {
			children[child.XMLName.Local] = childMap[child.XMLName.Local]
		}
	}

	xmlMsg[node.XMLName.Local] = children
	return xmlMsg
}

// The function MapToJSON converts map[string]interface{} to JSON string.
func mapToJSON(data map[string]interface{}) (string, error) {
	jsonData, err := json.Marshal(data)
	if err != nil {
		return "", err
	}
	return string(jsonData), nil
}

func StructToJSON(v interface{}) (string, error) {
	jsonData, err := json.MarshalIndent(v, "", "  ")
	if err != nil {
		return "", err
	}
	return string(jsonData), nil
}

// ConvertConfigDataToVisitorConfig converts simple ConfigData format to VisitorConfigData format
func ConvertConfigDataToVisitorConfig(protocolName string, configData map[string]interface{}) (*VisitorConfig, error) {
	// Extract topic to determine operation info and serialized format
	topicStr, ok := configData["topic"].(string)
	if !ok {
		return nil, errors.New("topic field missing or not a string")
	}

	parts := strings.Split(topicStr, "/")
	if len(parts) < 3 {
		return nil, errors.New("topic format is invalid, must have at least three parts")
	}

	deviceInfo := strings.Join(parts[:len(parts)-2], "/")
	operationStr := parts[len(parts)-2]
	formatStr := parts[len(parts)-1]

	// Convert operation string to enum
	var operationInfo OperationInfoType
	switch operationStr {
	case "deviceinfo":
		operationInfo = DEVICEINfO
	case "update":
		operationInfo = UPDATE
	case "setsinglevalue":
		operationInfo = SETSINGLEVALUE
	case "getsinglevalue":
		operationInfo = GETSINGLEVALUE
	default:
		operationInfo = GETSINGLEVALUE // default
	}

	// Convert format string to enum
	var serializedFormat SerializedFormatType
	switch formatStr {
	case "json":
		serializedFormat = JSON
	case "yaml":
		serializedFormat = YAML
	case "xml":
		serializedFormat = XML
	default:
		serializedFormat = JSON // default
	}

	// Extract clientID
	clientID, _ := configData["clientID"].(string)

	return &VisitorConfig{
		ProtocolName: protocolName,
		VisitorConfigData: VisitorConfigData{
			DataType:         "string",
			ClientID:         clientID,
			DeviceInfo:       deviceInfo,
			OperationInfo:    operationInfo,
			SerializedFormat: serializedFormat,
			ParsedMessage:    make(map[string]interface{}),
		},
	}, nil
}

/* --------------------------------------------------------------------------------------- */
