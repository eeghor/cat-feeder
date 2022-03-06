import unittest
import datetime
from cat_feeder import CatFeeder


class testSomething(unittest.TestCase):
    def test_id_check(self):

        self.assertFalse(CatFeeder()._is_valid_id(some_id="ywib11e34r32"))
        self.assertTrue(
            CatFeeder()._is_valid_id(some_id="a4d2cc9d-367a-4ec1-882a-618f34195aa1")
        )

    def test_count_of_recommended_posts(self):
        cat_feeder = CatFeeder()
        recommended_post_list = cat_feeder.feed(
            uid="eaf1456d-fe19-460b-9b98-ef860fe7b228", current_time=datetime.datetime.utcnow()
        ).posts_to_show
        self.assertEqual(len(recommended_post_list), cat_feeder.max_posts_to_show)


if __name__ == "__main__":
    unittest.main()
