package com.capstone.api;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;

/**
 * Person C — Spring Boot API Entry Point (Weeks 1-2)
 *
 * Responsibilities:
 *   - POST /ingest   → publish document to Kafka
 *   - POST /search   → call Python agent service and return answer
 *   - GET  /health   → liveness probe for K8s / Docker healthcheck
 */
@SpringBootApplication
public class ApiApplication {
    public static void main(String[] args) {
        SpringApplication.run(ApiApplication.class, args);
    }
}
