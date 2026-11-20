# Database Indexing Strategies: B-Trees vs LSM-Trees

Modern relational and analytical storage engines employ distinct index architectures optimized for read vs write workloads.

## B+ Tree Architecture

B+ Trees are balanced search trees where:
- All values and records are stored exclusively in the leaf nodes.
- Internal nodes contain only routing keys and child pointers, maximizing fanout and minimizing tree height.
- Leaf nodes are linked sequentially as a doubly linked list, enabling fast range scans in $O(\log N + K)$ time.
- Standard relational databases (PostgreSQL, SQLite, DuckDB) use B+ Trees for primary and secondary indexes.

## Log-Structured Merge (LSM) Trees

LSM-Trees optimize for write-intensive operations:
- Incoming writes append sequentially to an in-memory MemTable (typically a SkipList or Red-Black Tree) and a write-ahead log (WAL).
- When the MemTable fills up, it is flushed to disk as an immutable Sorted String Table (SSTable).
- Background compaction merges overlapping SSTables and eliminates tombstones.
- LSM-Trees trade read amplification for ultra-high write throughput.
