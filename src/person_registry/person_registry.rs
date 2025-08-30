use dlib_wrappers::face_encoding::FaceEncoding;

pub trait PersonRegistry {
    fn get(&self, encoding: &FaceEncoding) -> Option<&String>;
    fn add(&mut self, encoding: FaceEncoding, name: String);
}
