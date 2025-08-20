package device

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"time"

	"k8s.io/klog/v2"

	dmiapi "github.com/kubeedge/kubeedge/pkg/apis/dmi/v1beta1"
	"github.com/kubeedge/mapper-framework/pkg/common"
	"github.com/kubeedge/mapper-framework/pkg/grpcclient"
	"github.com/kubeedge/mapper-framework/pkg/util/parse"
	"github.com/kubeedge/mqtt/driver"
	"github.com/kubeedge/mqtt/pkg/metrics"
)

type TwinData struct {
	DeviceName      string
	DeviceNamespace string
	Client          *driver.CustomizedClient
	Name            string
	Type            string
	ObservedDesired common.TwinProperty
	VisitorConfig   *driver.VisitorConfig
	Topic           string
	Results         interface{}
	CollectCycle    time.Duration
	ReportToCloud   bool
}

func (td *TwinData) GetPayLoad() ([]byte, error) {
	var err error
	td.VisitorConfig.VisitorConfigData.DataType = strings.ToLower(td.VisitorConfig.VisitorConfigData.DataType)
	td.Results, err = td.Client.GetDeviceData(td.VisitorConfig)
	if err != nil {
		return nil, fmt.Errorf("get device data failed: %v", err)
	}

	// Extract the specific property value from the device data map
	var propertyValue interface{}
	klog.V(3).Infof("TwinData.GetPayLoad() for %s.%s: td.Results type=%T", td.DeviceName, td.Name, td.Results)
	if deviceData, ok := td.Results.(driver.MQTTDeviceData); ok {
		// Try to get the specific property by name
		if value, exists := deviceData[td.Name]; exists {
			propertyValue = value
			klog.V(2).Infof("Twin data extracted %s: %v", td.Name, value)
		} else {
			// If property not found, return a default value or error indication
			propertyValue = ""
			klog.V(1).Infof("Property %s not found in device data, using empty value", td.Name)
		}
	} else {
		// Fallback if not the expected type
		propertyValue = td.Results
		klog.V(1).Infof("Twin data type assertion failed, using fallback")
	}

	sData, err := common.ConvertToString(propertyValue)
	if err != nil {
		klog.Errorf("Failed to convert %s %s value as string : %v", td.DeviceName, td.Name, err)
		return nil, err
	}
	if len(sData) > 30 {
		klog.V(4).Infof("Get %s : %s ,value is %s......", td.DeviceName, td.Name, sData[:30])
	} else {
		klog.V(4).Infof("Get %s : %s ,value is %s", td.DeviceName, td.Name, sData)
	}
	var payload []byte
	klog.V(3).Infof("Twin update topic: %s", td.Topic)
	if strings.Contains(td.Topic, "$hw") {
		if payload, err = common.CreateMessageTwinUpdate(td.Name, td.Type, sData, td.ObservedDesired.Value); err != nil {
			return nil, fmt.Errorf("create message twin update failed: %v", err)
		}
		klog.V(3).Infof("Created twin update payload: %s", string(payload))
	} else {
		if payload, err = common.CreateMessageData(td.Name, td.Type, sData); err != nil {
			return nil, fmt.Errorf("create message data failed: %v", err)
		}
		klog.V(3).Infof("Created data payload: %s", string(payload))
	}
	return payload, nil
}

func (td *TwinData) PushToEdgeCore() {
	payload, err := td.GetPayLoad()
	if err != nil {
		klog.Errorf("twindata %s unmarshal failed, err: %s", td.Name, err)
		return
	}

	var msg common.DeviceTwinUpdate
	if err = json.Unmarshal(payload, &msg); err != nil {
		klog.Errorf("twindata %s unmarshal failed, err: %s", td.Name, err)
		return
	}

	twins := parse.ConvMsgTwinToGrpc(msg.Twin)

	var rdsr = &dmiapi.ReportDeviceStatusRequest{
		DeviceName:      td.DeviceName,
		DeviceNamespace: td.DeviceNamespace,
		ReportedDevice: &dmiapi.DeviceStatus{
			Twins: twins,
			//State: "OK",
		},
	}

	if err := grpcclient.ReportDeviceStatus(rdsr); err != nil {
		klog.Errorf("fail to report device status of %s with err: %+v", rdsr.DeviceName, err)
	} else {
		// Update Prometheus metrics after successful twin update
		td.updatePrometheusMetrics(&msg)
	}
}

// updatePrometheusMetrics updates Prometheus metrics based on twin data
func (td *TwinData) updatePrometheusMetrics(msg *common.DeviceTwinUpdate) {
	if msg.Twin == nil {
		return
	}

	metricsManager := metrics.GetManager()

	// Extract the reported value from the twin data
	// The msg.Twin contains property name -> *MsgTwin mapping
	if msgTwin, exists := msg.Twin[td.Name]; exists && msgTwin != nil {
		// Check if there's an Actual value (reported value)
		if msgTwin.Actual != nil && msgTwin.Actual.Value != nil {
			valueStr := *msgTwin.Actual.Value
			metricsManager.UpdateDeviceProperty(td.DeviceName, td.DeviceNamespace, td.Name, valueStr)
			klog.V(3).Infof("Updated Prometheus metric for device %s property %s: %s", td.DeviceName, td.Name, valueStr)
		}
	}
}

func (td *TwinData) Run(ctx context.Context) {
	klog.V(2).Infof("Starting twin data collection for %s.%s (cycle: %v)", td.DeviceName, td.Name, td.CollectCycle)
	if !td.ReportToCloud {
		klog.Warningf("Twin reporting disabled for %s.%s", td.DeviceName, td.Name)
		return
	}
	if td.CollectCycle == 0 {
		td.CollectCycle = common.DefaultCollectCycle
		klog.V(2).Infof("Using default collect cycle for %s.%s: %v", td.DeviceName, td.Name, td.CollectCycle)
	}

	ticker := time.NewTicker(td.CollectCycle)
	for {
		select {
		case <-ticker.C:
			klog.V(3).Infof("Twin data ticker fired for %s.%s", td.DeviceName, td.Name)
			td.PushToEdgeCore()
			// Add a small delay to prevent rate limiting
			time.Sleep(100 * time.Millisecond)
		case <-ctx.Done():
			klog.V(2).Infof("Twin data collection stopped for %s.%s", td.DeviceName, td.Name)
			return
		}
	}
}
