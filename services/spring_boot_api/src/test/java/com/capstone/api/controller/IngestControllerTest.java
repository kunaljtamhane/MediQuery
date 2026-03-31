package com.capstone.api.controller;

import com.capstone.api.service.KafkaProducerService;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.autoconfigure.web.servlet.WebMvcTest;
import org.springframework.boot.test.mock.mockito.MockBean;
import org.springframework.http.MediaType;
import org.springframework.test.web.servlet.MockMvc;

import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.never;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.post;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.*;

@WebMvcTest(IngestController.class)
class IngestControllerTest {

    @Autowired
    private MockMvc mockMvc;

    @MockBean
    private KafkaProducerService kafkaProducer;

    private static final String VALID_BODY = """
            {
              "docId": "arxiv-001",
              "title": "Attention Is All You Need",
              "text": "We propose a new network architecture, the Transformer.",
              "authors": "Vaswani et al.",
              "publishedDate": "2017-06-12",
              "arxivUrl": "https://arxiv.org/abs/1706.03762"
            }
            """;

    // ── Happy path ────────────────────────────────────────────────────────────

    @Test
    void ingest_validRequest_returns202Accepted() throws Exception {
        mockMvc.perform(post("/ingest")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content(VALID_BODY))
                .andExpect(status().isAccepted());
    }

    @Test
    void ingest_validRequest_returnsQueuedStatus() throws Exception {
        mockMvc.perform(post("/ingest")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content(VALID_BODY))
                .andExpect(jsonPath("$.status").value("queued"));
    }

    @Test
    void ingest_validRequest_returnsDocId() throws Exception {
        mockMvc.perform(post("/ingest")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content(VALID_BODY))
                .andExpect(jsonPath("$.doc_id").value("arxiv-001"));
    }

    @Test
    void ingest_validRequest_publishesToKafka() throws Exception {
        mockMvc.perform(post("/ingest")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content(VALID_BODY))
                .andExpect(status().isAccepted());

        verify(kafkaProducer).publishDocument(eq("arxiv-001"), anyString());
    }

    // ── Validation failures ───────────────────────────────────────────────────

    @Test
    void ingest_missingDocId_returns400() throws Exception {
        String body = """
                {
                  "title": "Some Title",
                  "text": "Some text content here."
                }
                """;
        mockMvc.perform(post("/ingest")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content(body))
                .andExpect(status().isBadRequest());
    }

    @Test
    void ingest_missingTitle_returns400() throws Exception {
        String body = """
                {
                  "docId": "arxiv-001",
                  "text": "Some text content here."
                }
                """;
        mockMvc.perform(post("/ingest")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content(body))
                .andExpect(status().isBadRequest());
    }

    @Test
    void ingest_missingText_returns400() throws Exception {
        String body = """
                {
                  "docId": "arxiv-001",
                  "title": "Some Title"
                }
                """;
        mockMvc.perform(post("/ingest")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content(body))
                .andExpect(status().isBadRequest());
    }

    @Test
    void ingest_blankDocId_returns400() throws Exception {
        String body = """
                {
                  "docId": "   ",
                  "title": "Some Title",
                  "text": "Some text content here."
                }
                """;
        mockMvc.perform(post("/ingest")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content(body))
                .andExpect(status().isBadRequest());
    }

    @Test
    void ingest_validationFails_neverPublishesToKafka() throws Exception {
        String body = """
                {
                  "title": "Some Title"
                }
                """;
        mockMvc.perform(post("/ingest")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content(body))
                .andExpect(status().isBadRequest());

        verify(kafkaProducer, never()).publishDocument(anyString(), anyString());
    }
}
