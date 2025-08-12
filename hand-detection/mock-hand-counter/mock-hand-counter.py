#!/usr/bin/env python3
"""
Simple Mock Hand Counter - Cycles through 0, 1, 2, 3 hands every 2 seconds
Publishes data to MQTT topic
"""

import time
import json
import os
from datetime import datetime

try:
    import paho.mqtt.client as mqtt
    MQTT_AVAILABLE = True
except ImportError:
    print("Warning: paho-mqtt not installed. MQTT publishing disabled.")
    MQTT_AVAILABLE = False

def load_config(config_file=None):
    """Load configuration from JSON file"""
    # Use environment variable if set, otherwise use default
    if config_file is None:
        config_file = os.getenv("CONFIG_PATH", "config.json")
    
    try:
        with open(config_file, 'r') as f:
            config = json.load(f)
            print(f"Loaded configuration from: {config_file}")
            return config
    except FileNotFoundError:
        print(f"Warning: Config file '{config_file}' not found. Using default values.")
        return {
            "mqtt": {
                "broker_host": "localhost",
                "broker_port": 1883,
                "topic": "hand-detection/count",
                "timeout": 60
            },
            "simulation": {
                "update_interval": 2.0,
                "max_count": 3
            }
        }
    except json.JSONDecodeError as e:
        print(f"Error parsing config file: {e}. Using default values.")
        # Return default config instead of recursion
        return {
            "mqtt": {
                "broker_host": "localhost",
                "broker_port": 1883,
                "topic": "hand-detection/count",
                "timeout": 60
            },
            "simulation": {
                "update_interval": 2.0,
                "max_count": 3
            }
        }

class MQTTPublisher:
    def __init__(self, config):
        self.broker_host = config["mqtt"]["broker_host"]
        self.broker_port = config["mqtt"]["broker_port"]
        self.topic = config["mqtt"]["topic"]
        self.timeout = config["mqtt"]["timeout"]
        self.client = None
        self.connected = False
        
        if MQTT_AVAILABLE:
            self.setup_mqtt()
    
    def setup_mqtt(self):
        """Initialize MQTT client"""
        self.client = mqtt.Client()
        self.client.on_connect = self.on_connect
        self.client.on_disconnect = self.on_disconnect
        
        try:
            print(f"Connecting to MQTT broker at {self.broker_host}:{self.broker_port}")
            self.client.connect(self.broker_host, self.broker_port, self.timeout)
            self.client.loop_start()
        except Exception as e:
            print(f"Failed to connect to MQTT broker: {e}")
            print("Continuing without MQTT...")
    
    def on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            self.connected = True
            print(f"Connected to MQTT broker. Publishing to topic: {self.topic}")
        else:
            print(f"Failed to connect to MQTT broker. Return code: {rc}")
    
    def on_disconnect(self, client, userdata, rc):
        self.connected = False
        print("Disconnected from MQTT broker")
    
    def publish(self, hand_count):
        """Publish hand count data to MQTT topic"""
        if not MQTT_AVAILABLE or not self.connected:
            return
        
        # Create message payload in the required format
        payload = {
            "handsraised": str(hand_count),
            "status": "online"
        }
        
        try:
            self.client.publish(self.topic, json.dumps(payload))
            print(f"Published to MQTT: {payload}")
        except Exception as e:
            print(f"Failed to publish to MQTT: {e}")
    
    def disconnect(self):
        """Disconnect from MQTT broker"""
        if self.client and self.connected:
            self.client.loop_stop()
            self.client.disconnect()

def main():
    # Load configuration
    config = load_config()
    
    print("Simple Mock Hand Counter Started!")
    print(f"Cycles through 0 to {config['simulation']['max_count']} hands every {config['simulation']['update_interval']} seconds")
    print(f"Publishes to MQTT broker: {config['mqtt']['broker_host']}:{config['mqtt']['broker_port']}")
    print(f"Topic: {config['mqtt']['topic']}")
    print("Press Ctrl+C to stop")
    print("-" * 50)
    
    # Initialize MQTT publisher with config
    mqtt_publisher = MQTTPublisher(config)
    
    count = 0
    max_count = config['simulation']['max_count']
    update_interval = config['simulation']['update_interval']
    
    try:
        while True:
            print(f"Hands raised: {count}")
            
            # Publish to MQTT
            mqtt_publisher.publish(count)
            
            time.sleep(update_interval)
            count = (count + 1) % (max_count + 1)  # Cycle through 0 to max_count
            
    except KeyboardInterrupt:
        print("\nMock simulation stopped.")
        mqtt_publisher.disconnect()

if __name__ == "__main__":
    main()
