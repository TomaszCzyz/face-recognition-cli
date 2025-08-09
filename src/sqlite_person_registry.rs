use crate::person_registry::PersonRegistry;
use dlib_wrappers::face_encoding::FaceEncoding;

struct SqlitePersonRegistry {}

impl PersonRegistry for SqlitePersonRegistry {
    fn get(&self, encoding: &FaceEncoding) -> Option<&String> {
        todo!()
    }

    fn add(&mut self, encoding: FaceEncoding, name: String) {
        todo!()
    }

    fn save(&self) {
        todo!()
    }
}
