CREATE TABLE ProcessedFiles
(
    Id          INTEGER PRIMARY KEY AUTOINCREMENT,
    Hash        TEXT UNIQUE,
    Path        TEXT,
    ProcessedAt DATETIME DEFAULT (CURRENT_TIMESTAMP)
);

CREATE TABLE Faces
(
    FileId       INTEGER,
    FaceEncoding float[128] check ( typeof(FaceEncoding) == 'blob' and vec_length(FaceEncoding) == 128 ),
    Id           INTEGER PRIMARY KEY AUTOINCREMENT,
    RectLeft     INTEGER NOT NULL,
    RectTop      INTEGER NOT NULL,
    RectRight    INTEGER NOT NULL,
    RectBottom   INTEGER NOT NULL,
    CreatedAt    DATETIME DEFAULT (CURRENT_TIMESTAMP)
);
