CREATE TABLE ProcessedFiles
(
    Id          INTEGER PRIMARY KEY AUTOINCREMENT,
    Hash        TEXT UNIQUE,
    Path        TEXT,
    ProcessedAt DATETIME DEFAULT (CURRENT_TIMESTAMP)
);
