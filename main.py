import face_categorization as fc
import faceMesh as fm

if __name__ == "__main__":
    face_shape = fc.get_face_shape()
    fm.applyContour(face_shape)