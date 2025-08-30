CREATE VIRTUAL TABLE FaceEncodings using vec0
(
    id integer primary key,
    face_encoding float [128]
);


-- File --1:n--> Face

-- Face --1:1--> Encoding
-- Face --1:1--> Location

-- Person --1:n--> Face

CREATE TABLE Faces
(
    Id          INTEGER PRIMARY KEY AUTOINCREMENT,
    File_Id     INTEGER,
    Encoding_Id INTEGER,
    Location_Id INTEGER,
    CreatedAt   DATETIME DEFAULT (CURRENT_TIMESTAMP)
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

-- CREATE TABLE Persons
-- (
--     Id   INTEGER PRIMARY KEY AUTOINCREMENT,
--     Name TEXT
-- );
--
-- CREATE TABLE PersonsFaces
-- (
--     Id   INTEGER PRIMARY KEY AUTOINCREMENT,
--     Name TEXT
-- );
