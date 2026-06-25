import unittest

from pipeline import associate_faces_with_helmets
from schemas import Detection, FaceIdentity


class AssociationTest(unittest.TestCase):
    def test_face_is_marked_helmet_when_matching_head_has_overlapping_helmet(self):
        faces = [FaceIdentity(name="张三", confidence=0.82, box=(45, 40, 95, 105))]
        detections = [
            Detection(class_name="head", confidence=0.91, box=(40, 30, 100, 110)),
            Detection(class_name="helmet", confidence=0.88, box=(35, 20, 105, 65)),
        ]

        results = associate_faces_with_helmets(faces, detections)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].name, "张三")
        self.assertTrue(results[0].has_helmet)

    def test_face_is_marked_no_helmet_when_matching_head_has_no_helmet_nearby(self):
        faces = [FaceIdentity(name="李四", confidence=0.78, box=(200, 80, 250, 145))]
        detections = [
            Detection(class_name="head", confidence=0.9, box=(195, 70, 255, 155)),
            Detection(class_name="helmet", confidence=0.93, box=(20, 20, 80, 70)),
        ]

        results = associate_faces_with_helmets(faces, detections)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].name, "李四")
        self.assertFalse(results[0].has_helmet)

    def test_unknown_face_without_matching_head_uses_face_region_for_helmet_check(self):
        faces = [FaceIdentity(name="unknown", confidence=0.0, box=(60, 50, 110, 120))]
        detections = [Detection(class_name="helmet", confidence=0.86, box=(50, 35, 120, 80))]

        results = associate_faces_with_helmets(faces, detections)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].name, "unknown")
        self.assertTrue(results[0].has_helmet)


if __name__ == "__main__":
    unittest.main()
