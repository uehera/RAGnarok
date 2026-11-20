# API Gateway Architecture and Rate Limiting

The API Gateway is the single point of entry for all incoming client traffic in a microservices ecosystem. It handles reverse proxy routing, authentication, request validation, and rate limiting.

## Token Bucket Algorithm

The token bucket algorithm is a widely adopted traffic shaping algorithm:
- Tokens are added to a bucket at a fixed fill rate $r$ tokens per second.
- The bucket has a maximum capacity $b$ tokens.
- Each incoming request consumes one or more tokens from the bucket.
- If insufficient tokens remain when a request arrives, the request is rejected with HTTP 429 Too Many Requests.
- The token bucket allows bursts of up to $b$ requests while enforcing a long-term average rate limit of $r$.

## Circuit Breakers

Circuit breakers prevent cascading service failures:
- Closed: Traffic flows normally to downstream microservices.
- Open: When failure rate exceeds threshold (e.g. 50% errors over 10 seconds), subsequent requests fail immediately without calling the unhealthy service.
- Half-Open: After a cooldown window (e.g. 30 seconds), a canary probe request is permitted through to test downstream health.
