import pandas as pd  # type: ignore
import string
import re
from typing import List, Any, Dict, DefaultDict, Literal, Set
import datetime
import spacy
from itertools import combinations, chain
from collections import defaultdict
from gender import GenderDetector  # type: ignore

nlp = spacy.load("en_core_web_sm")
gd = GenderDetector()


class CatFeeder:

    AGE_GROUPS = {
        g: i
        for i, g in enumerate(
            [(from_, to_) for from_, to_ in zip(range(0, 100, 10), range(9, 100, 10))],
            1,
        )
    }
    DAYS_IN_A_YEAR = 365

    def __init__(
        self,
        user_details_file: str = "data/users.csv",
        user_interests_file: str = "data/interest[1].csv",
        posts_file: str = "data/posts.csv",
    ) -> None:

        self.users = pd.read_csv(user_details_file, parse_dates=["dob"])
        self.interests = pd.read_csv(user_interests_file)
        self.posts = pd.read_csv(posts_file, parse_dates=["post_time"])

    def find_out_more_about_users(self) -> "CatFeeder":

        """
        find out users age and gender; also an age group they fall into

        Parameters
        ----------
        none

        Returns
        -------
        self
        """

        self.users["gender"] = (
            self.users["first_name"] + " " + self.users["family_name"]
        ).apply(lambda x: gd.get_gender(x).gender)
        self.users["age_years"] = (
            self.users["dob"]
            .apply(
                lambda x: (datetime.datetime.utcnow() - x).days / self.DAYS_IN_A_YEAR
            )
            .round()
        )
        self.users["age_group"] = self.users["age_years"].apply(
            lambda x: [
                self.AGE_GROUPS[age_group]
                for age_group in self.AGE_GROUPS
                if (age_group[0] <= x <= age_group[1])
            ].pop()
        )

        return self

    def _map_post_hashtags(self, to: Literal["uid", "post_id"]) -> Dict[str, Set[str]]:

        return (
            self.posts[[to, "hashtags"]]
            .groupby(to)
            .agg(set)["hashtags"]
            .apply(
                lambda x: {
                    c.strip().lower()
                    for c in re.split(r"[\{\[\]\},\'\";]", str(x))
                    if c.strip()
                }
            )
            .to_dict()
        )

    def _map_users_and_posts_to_tags(self) -> "CatFeeder":

        self.interests = self.interests.groupby("uid").agg(set).to_dict()["interest"]
        self.users_write_about = self._map_post_hashtags(to="uid")
        self.posts_are_about = self._map_post_hashtags(to="post_id")

        return self

    def calculate_tag_similarity(
        self, set_of_tags: DefaultDict[str, Set[str]], normalize: bool = False
    ) -> DefaultDict[str, DefaultDict[str, float]]:

        """
        calculate similarity between sets of tags associated with certain IDs

        Parameters
        ----------
        set_of_tags
            is a dictionary of the form {ID: {tag1, tag2, ..}}
        normalize:
            normalize the output if True

        Returns
        -------
        tag_similarities
            a dictionary of the form {ID1: {ID2: {similarity score between ID1 and ID2}}}

        """

        tag_similarities: DefaultDict[str, DefaultDict[str, float]] = defaultdict(
            lambda: defaultdict(float)
        )

        for (id1, id2) in combinations(set_of_tags, 2):
            if id1_tags := set_of_tags.get(id1, None):
                if id2_tags := set_of_tags.get(id2, None):
                    tag_similarities[id1][id2] = len(id1_tags & id2_tags)
                    tag_similarities[id2][id1] = tag_similarities[id1][id2]

        if normalize:
            tag_similarities = self._normalize_nested_dictionary_values(
                tag_similarities
            )

        return tag_similarities

    def _normalize_nested_dictionary_values(
        self, dictionary: DefaultDict[str, DefaultDict[str, float]]
    ) -> DefaultDict[str, DefaultDict[str, float]]:

        largest_value = max(
            [
                max(nested_dictionary.values())
                for nested_dictionary in dictionary.values()
            ]
        )

        for key_ in dictionary:
            for nested_key_ in dictionary[key_]:
                dictionary[key_][nested_key_] = (
                    dictionary[key_][nested_key_] / largest_value
                )

        return dictionary

    def _is_valid_post_id(self, post_id: Any) -> bool:

        """
        check if the supplied argument looks like a legit post ID

        Parameters
        ----------
        post_id
            a string that could be a valid post ID

        Returns
        -------
        True or False

        """
        return len(str(post_id).split("-")) == 5

    def get_post_content_similarity(
        self, parts_of_speech: List[Literal["NOUN", "VERB"]] = ["NOUN"]
    ) -> DefaultDict[str, DefaultDict[str, float]]:
        """
        calculate similarity score for each two post texts

        Parameters
        ----------
        parts_of_speech
            ignore any other POS from the post texts except the ones on this list

        Returns
        -------
        True or False
        """

        self.posts["post_nouns"] = self.posts["text"].apply(
            lambda txt: {
                w.lemma_ for w in nlp(txt.lower()) if w.pos_ in parts_of_speech
            }
        )

        return self.calculate_tag_similarity(
            self.posts[["post_id", "post_nouns"]]
            .groupby("post_id")
            .agg(lambda x: set(chain.from_iterable(x)))
            .to_dict()["post_nouns"],
            normalize=True,
        )

    def get_demographic_similarity(self) -> DefaultDict[str, DefaultDict[str, float]]:

        self.users["age_and_gender"] = (
            (self.users["age_group"].astype(str) + " " + self.users["gender"])
            .str.split()
            .apply(set)
        )

        return self.calculate_tag_similarity(
            self.users[["uid", "age_and_gender"]]
            .set_index("uid")
            .to_dict()["age_and_gender"],
            normalize=True,
        )

    def map_posts_to_their_parents(self) -> "CatFeeder":

        valid_post_ids = {
            pid for pid in self.posts["post_id"] if self._is_valid_post_id(pid)
        }
        print(f"valid post IDs: {len(valid_post_ids):,}")
        valid_parent_post_ids = {
            pid for pid in self.posts["parent_id"] if self._is_valid_post_id(pid)
        }
        print(f"valid parent post IDs: {len(valid_parent_post_ids):,}")
        self.post_id_to_parent_post_id = {
            _[0]: _[1]
            for _ in self.posts[self.posts["parent_id"].apply(self._is_valid_post_id)][
                ["post_id", "parent_id"]
            ].to_dict(orient="split")["data"]
        }

        return self

    def review_data(self) -> "CatFeeder":

        """
        review data to see if there's anything worth knowing

        Parameters
        ----------
        none

        Returns
        -------
        a dictionary {id1: {id2: {similarity between user1 and user2}}}

        """
        # are there any users that seem like the same person but appear with different IDs?
        users_with_multiple_uid = len(
            self.users.groupby(["first_name", "family_name", "dob"])
            .count()
            .query("uid > 1")
        )
        print(f"users with multiple UID: {users_with_multiple_uid:,}")

        # are there any posts with multiple parents?
        posts_with_many_parents = len(
            self.posts[["post_id", "parent_id"]]
            .groupby("post_id")
            .nunique()
            .query("parent_id > 1")
        )
        print(f"posts with many parents: {posts_with_many_parents:,}")

        return self
