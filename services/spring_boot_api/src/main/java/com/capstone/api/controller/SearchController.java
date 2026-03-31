package com.capstone.api.controller;

import jakarta.validation.Valid;
import jakarta.validation.constraints.NotBlank;
import lombok.Data;
import lombok.RequiredArgsConstructor;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.MediaType;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.reactive.function.client.WebClient;
import reactor.core.publisher.Flux;

import java.util.Map;

/**
 * Person C — Search Endpoint (Weeks 7-8)
 *
 * POST /search
 *   Body: { "query": "...", "session_id": "..." }
 *   Forwards to the Python agent service and streams the response back.
 *
 * Week 1-2: Returns a stub response.
 * Week 7-8: Calls agentServiceUrl once agents are built.
 */
@RestController
@RequiredArgsConstructor
public class SearchController {

    @Value("${agent.service.url:http://agents:8004}")
    private String agentServiceUrl;

    private final WebClient.Builder webClientBuilder;

    @PostMapping(value = "/search", produces = MediaType.TEXT_EVENT_STREAM_VALUE)
    public Flux<String> search(@Valid @RequestBody SearchRequest request) {
        return webClientBuilder
            .baseUrl(agentServiceUrl)
            .build()
            .post()
            .uri("/query")
            .contentType(MediaType.APPLICATION_JSON)
            .bodyValue(Map.of(
                "query", request.getQuery(),
                "session_id", request.getSessionId()
            ))
            .retrieve()
            .bodyToFlux(String.class);
    }

    // ── Request DTO ──────────────────────────────────────────────────────────

    @Data
    public static class SearchRequest {
        @NotBlank private String query;
        private String sessionId = "default";
    }
}
