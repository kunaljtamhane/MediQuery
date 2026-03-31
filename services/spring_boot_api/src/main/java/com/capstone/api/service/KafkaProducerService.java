package com.capstone.api.service;

import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.kafka.core.KafkaTemplate;
import org.springframework.stereotype.Service;

/**
 * Person C — Kafka Producer (Weeks 1-2)
 * Publishes document JSON to the "documents" topic.
 * The Python kafka_consumer service reads from this topic, chunks,
 * embeds, and indexes the document.
 */
@Slf4j
@Service
@RequiredArgsConstructor
public class KafkaProducerService {

    @Value("${kafka.topic.documents:documents}")
    private String documentsTopic;

    private final KafkaTemplate<String, String> kafkaTemplate;

    public void publishDocument(String docId, String documentJson) {
        kafkaTemplate.send(documentsTopic, docId, documentJson)
            .whenComplete((result, ex) -> {
                if (ex != null) {
                    log.error("Failed to publish doc_id={} to Kafka: {}", docId, ex.getMessage());
                } else {
                    log.info("Published doc_id={} to topic={} partition={} offset={}",
                        docId, documentsTopic,
                        result.getRecordMetadata().partition(),
                        result.getRecordMetadata().offset());
                }
            });
    }
}
