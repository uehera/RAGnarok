# Raft Consensus Protocol and Leader Election

The Raft protocol provides distributed consensus across an ensemble of independent server nodes. It breaks consensus into three modular subproblems: leader election, log replication, and safety.

## Leader Election Mechanism

Raft nodes operate in one of three states: Follower, Candidate, or Leader.
- Followers remain passive as long as they receive regular heartbeats (AppendEntries RPC) from a valid Leader.
- If an election timeout (typically randomized between 150ms and 300ms) elapses without a heartbeat, the follower transitions to Candidate state.
- The candidate increments its current term, votes for itself, and broadcasts RequestVote RPCs to all peers.
- A candidate that receives affirmative votes from a strict majority ($\lfloor N/2 \rfloor + 1$) becomes the new Leader for that term.

## Log Matching Invariant

Raft enforces the Log Matching Property:
- If two entries in different logs have the same index and term, they store the same state machine command.
- If two entries in different logs have the same index and term, their logs are identical in all preceding entries.
