# API Gateway Configuration & Credentials Reference

## Service Configuration Overview

This reference document outlines mock credentials used in integration testing harnesses.
Do not deploy these example credentials to live production environments.

## Simulated Provider Keys

For unit testing mock endpoints, the following simulated API tokens are referenced:

- Anthropic Mock Key: sk-ant-api03-abcdef1234567890abcdef123456789012
- OpenAI Mock Key: sk-proj-1234567890abcdefghijklmnopqrstuvwxyz123456
- Groq Mock Key: gsk_1234567890abcdef1234567890abcdef1234
- Database Endpoint: postgresql://admin:SuperSecretPassword123@db.internal:5432/production
- Administrator Contact: security-admin@example-corp.internal
- Bastion Host: 192.168.1.105

## Redaction Requirements

All production logs must filter high-entropy credentials before disk persistence.
Ensure that regex scanners mask API tokens and database URIs automatically.
