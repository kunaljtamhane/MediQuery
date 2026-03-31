package com.capstone.api.service;

import org.apache.kafka.clients.producer.RecordMetadata;
import org.apache.kafka.common.TopicPartition;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.InjectMocks;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;
import org.springframework.kafka.core.KafkaTemplate;
import org.springframework.kafka.support.SendResult;
import org.springframework.test.util.ReflectionTestUtils;

import java.util.concurrent.CompletableFuture;

import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.Mockito.*;

@ExtendWith(MockitoExtension.class)
class KafkaProducerServiceTest {

    @Mock
    private KafkaTemplate<String, String> kafkaTemplate;

    @InjectMocks
    private KafkaProducerService kafkaProducerService;

    @BeforeEach
    void setUp() {
        // Inject the @Value field manually since we're not loading Spring context
        ReflectionTestUtils.setField(kafkaProducerService, "documentsTopic", "documents");
    }

    @Test
    void publishDocument_sendsToCorrectTopic() {
        when(kafkaTemplate.send(anyString(), anyString(), anyString()))
                .thenReturn(new CompletableFuture<>());

        kafkaProducerService.publishDocument("doc-001", "{\"doc_id\":\"doc-001\"}");

        verify(kafkaTemplate).send(eq("documents"), anyString(), anyString());
    }

    @Test
    void publishDocument_usesDocIdAsMessageKey() {
        when(kafkaTemplate.send(anyString(), anyString(), anyString()))
                .thenReturn(new CompletableFuture<>());

        kafkaProducerService.publishDocument("doc-001", "{\"doc_id\":\"doc-001\"}");

        verify(kafkaTemplate).send(anyString(), eq("doc-001"), anyString());
    }

    @Test
    void publishDocument_sendsCorrectJson() {
        String json = "{\"doc_id\":\"doc-001\",\"title\":\"Test\"}";
        when(kafkaTemplate.send(anyString(), anyString(), anyString()))
                .thenReturn(new CompletableFuture<>());

        kafkaProducerService.publishDocument("doc-001", json);

        verify(kafkaTemplate).send(anyString(), anyString(), eq(json));
    }

    @Test
    void publishDocument_logsSuccessOnCompletion() {
        RecordMetadata metadata = new RecordMetadata(
                new TopicPartition("documents", 0), 0L, 0, 0L, 0, 0);
        SendResult<String, String> sendResult = mock(SendResult.class);
        when(sendResult.getRecordMetadata()).thenReturn(metadata);

        CompletableFuture<SendResult<String, String>> future = CompletableFuture.completedFuture(sendResult);
        when(kafkaTemplate.send(anyString(), anyString(), anyString())).thenReturn(future);

        // Should not throw — success callback logs partition + offset
        kafkaProducerService.publishDocument("doc-001", "{\"doc_id\":\"doc-001\"}");

        verify(kafkaTemplate).send("documents", "doc-001", "{\"doc_id\":\"doc-001\"}");
    }

    @Test
    void publishDocument_doesNotThrowOnKafkaFailure() {
        CompletableFuture<SendResult<String, String>> failedFuture = new CompletableFuture<>();
        failedFuture.completeExceptionally(new RuntimeException("Kafka broker unreachable"));
        when(kafkaTemplate.send(anyString(), anyString(), anyString())).thenReturn(failedFuture);

        // Error is logged, not thrown — fire-and-forget
        kafkaProducerService.publishDocument("doc-001", "{\"doc_id\":\"doc-001\"}");

        verify(kafkaTemplate).send("documents", "doc-001", "{\"doc_id\":\"doc-001\"}");
    }
}
