package com.capstone.api.controller;

import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.autoconfigure.web.servlet.WebMvcTest;
import org.springframework.test.web.servlet.MockMvc;

import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.get;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.*;

@WebMvcTest(HealthController.class)
class HealthControllerTest {

    @Autowired
    private MockMvc mockMvc;

    @Test
    void health_returns200() throws Exception {
        mockMvc.perform(get("/health"))
                .andExpect(status().isOk());
    }

    @Test
    void health_returnsStatusOk() throws Exception {
        mockMvc.perform(get("/health"))
                .andExpect(jsonPath("$.status").value("ok"));
    }

    @Test
    void health_returnsServiceName() throws Exception {
        mockMvc.perform(get("/health"))
                .andExpect(jsonPath("$.service").value("spring-boot-api"));
    }
}
