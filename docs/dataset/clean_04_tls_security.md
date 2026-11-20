# TLS 1.3 Handshake Protocol and Forward Secrecy

Transport Layer Security (TLS) 1.3 significantly streamlines the cryptographic handshake compared to TLS 1.2, reducing handshake latency to a single round-trip time (1-RTT).

## Ephemeral Diffie-Hellman Key Exchange

TLS 1.3 achieves Perfect Forward Secrecy (PFS) by mandating ephemeral key exchange:
- Client generates an ephemeral private/public keypair $(a, g^a \pmod p)$ and sends $g^a$ in the ClientHello message.
- Server generates its ephemeral keypair $(b, g^b \pmod p)$ and sends $g^b$ in the ServerHello.
- Both parties compute the shared secret $(g^{ab} \pmod p)$.
- Because ephemeral keys are discarded immediately after session derivation, compromising the server's long-term private key does not compromise past encrypted traffic.

## Zero Round-Trip Resumption (0-RTT)

Clients reconnecting to a previously visited server can transmit early application data in the first flight:
- 0-RTT data is encrypted using a Pre-Shared Key (PSK) derived from a prior session ticket.
- Warning: 0-RTT does not provide forward secrecy and is vulnerable to replay attacks unless protected by single-use ticket databases or client timestamp validation.
