-- File --1:n--> Face

-- Face --1:1--> Encoding
-- Face --1:1--> Location

-- Person --1:n--> Face

CREATE TABLE FaceEncodings
(
    Id            INTEGER PRIMARY KEY AUTOINCREMENT,
    FaceEncoding float[128]
        check ( typeof(FaceEncodings.FaceEncoding) == 'blob' and vec_length(FaceEncoding) == 128 )
);

CREATE TABLE Faces
(
    Id         INTEGER PRIMARY KEY AUTOINCREMENT,
    FileId     INTEGER,
    EncodingId INTEGER,
    LocationId INTEGER,
    CreatedAt  DATETIME DEFAULT (CURRENT_TIMESTAMP)
);

CREATE TABLE FaceLocations
(
    Id        INTEGER PRIMARY KEY AUTOINCREMENT,
    "Left"    INTEGER NOT NULL,
    Top       INTEGER NOT NULL,
    "Right"   INTEGER NOT NULL,
    Bottom    INTEGER NOT NULL,
    CreatedAt DATETIME DEFAULT (CURRENT_TIMESTAMP)
);
