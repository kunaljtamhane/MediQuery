package com.capstone.api.controller;

import com.capstone.api.service.KafkaProducerService;
import jakarta.validation.Valid;
import jakarta.validation.constraints.NotBlank;
import lombok.Data;
import lombok.RequiredArgsConstructor;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.Map;

/**
 * Person C — Document Ingestion Endpoint (Weeks 1-2)
 *
 * POST /ingest
 *   Body: { "doc_id": "...", "title": "...", "text": "...", "metadata": {} }
 *   Publishes to Kafka topic "documents".
 *   Python Kafka consumer (kafka_consumer service) picks it up and indexes it.
 */
@RestController
@RequiredArgsConstructor
public class IngestController {

    private final KafkaProducerService kafkaProducer;

    @PostMapping("/ingest")
    public ResponseEntity<Map<String, String>> ingest(@Valid @RequestBody IngestRequest request) {
        kafkaProducer.publishDocument(request.getDocId(), request.toJson());
        return ResponseEntity.accepted().body(Map.of(
            "status", "queued",
            "doc_id", request.getDocId()
        ));
    }

    // ── Request DTO ──────────────────────────────────────────────────────────

    @Data
    public static class IngestRequest {
        @NotBlank private String docId;
        @NotBlank private String title;
        @NotBlank private String text;
        private String authors = "";
        private String publishedDate = "";
        private String arxivUrl = "";

        public String toJson() {
            // Simple JSON serialisation — swap for Jackson ObjectMapper if preferred
            return String.format(
                "{\"doc_id\":\"%s\",\"title\":\"%s\",\"text\":\"%s\",\"authors\":\"%s\",\"published\":\"%s\",\"url\":\"%s\"}",
                docId, escape(title), escape(text), escape(authors), publishedDate, arxivUrl
            );
        }

        private String escape(String s) {
            return s == null ? "" : s.replace("\"", "\\\"").replace("\n", "\\n");
        }
    }
}
