package com.capstone.api.controller;

import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RestController;

import java.util.Map;

/**
 * Person C — Health endpoint (Weeks 1-2)
 * Required by Docker Compose healthcheck and Kubernetes liveness probe.
 */
@RestController
public class HealthController {

    @GetMapping("/health")
    public Map<String, String> health() {
        return Map.of("status", "ok", "service", "spring-boot-api");
    }
}
