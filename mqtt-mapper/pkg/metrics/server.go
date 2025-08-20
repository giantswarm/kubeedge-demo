package metrics

import (
	"context"
	"fmt"
	"net/http"
	"time"

	"github.com/prometheus/client_golang/prometheus/promhttp"
	"k8s.io/klog/v2"
)

// MetricsServer provides HTTP server for Prometheus metrics endpoint
type MetricsServer struct {
	server *http.Server
	port   int
}

// NewMetricsServer creates a new metrics server
func NewMetricsServer(port int) *MetricsServer {
	mux := http.NewServeMux()
	mux.Handle("/metrics", promhttp.Handler())

	// Add a health check endpoint
	mux.HandleFunc("/health", func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		w.Write([]byte("OK"))
	})

	server := &http.Server{
		Addr:    fmt.Sprintf(":%d", port),
		Handler: mux,
	}

	return &MetricsServer{
		server: server,
		port:   port,
	}
}

// Start starts the metrics server in a goroutine
func (ms *MetricsServer) Start() error {
	klog.Infof("Starting Prometheus metrics server on port %d", ms.port)

	go func() {
		if err := ms.server.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			klog.Errorf("Failed to start metrics server: %v", err)
		}
	}()

	// Give the server a moment to start
	time.Sleep(100 * time.Millisecond)
	klog.Infof("Metrics server started successfully at http://localhost:%d/metrics", ms.port)

	return nil
}

// Stop gracefully stops the metrics server
func (ms *MetricsServer) Stop() error {
	klog.Info("Stopping Prometheus metrics server...")

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	if err := ms.server.Shutdown(ctx); err != nil {
		klog.Errorf("Failed to gracefully shutdown metrics server: %v", err)
		return err
	}

	klog.Info("Metrics server stopped successfully")
	return nil
}
