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

pattern = re.compile(r"[\{\[\]\},\'\";]")


class CatFeeder:

    AGE_GROUPS = {
        g: i
        for i, g in enumerate(
            [(from_, to_) for from_, to_ in zip(range(0, 100, 10), range(9, 100, 10))],
            1,
        )
    }
    DAYS_IN_A_YEAR = 365
    HYPHENS_IN_ID = 4

    def __init__(
        self,
        user_details_file: str = "data/users.csv",
        user_interests_file: str = "data/interest[1].csv",
        posts_file: str = "data/posts.csv",
    ) -> None:

        self.users = pd.read_csv(user_details_file, parse_dates=["dob"])
        self.interests = pd.read_csv(user_interests_file)
        self.posts = pd.read_csv(posts_file, parse_dates=["post_time"])

    def _is_valid_id(self, post_id: Any) -> bool:

        """
        check if the supplied argument looks like a legit ID

        Parameters
        ----------
        post_id
            an argument that could be a valid ID

        Returns
        -------
        True if valid, False otherwise

        """
        return len(str(post_id).split("-")) == (self.HYPHENS_IN_ID + 1)

    def _map_post_hashtags(self, to: Literal["uid", "post_id"]) -> Dict[str, Set[str]]:

        """
        map post hashtags to either user IDs or post IDs

        Parameters
        ----------
        to:
            uid or post_id

        Returns
        -------
        dictionary of the form {ID: {hashtag1, hashtag2,...}}

        """

        return (
            self.posts[[to, "hashtags"]]
            .groupby(to)
            .agg(set)["hashtags"]
            .apply(
                lambda x: {
                    c.strip().lower() for c in pattern.split(str(x)) if c.strip()
                }
            )
            .to_dict()
        )

    def _map_users_and_posts_to_tags(self) -> "CatFeeder":

        self.users_write_about = self._map_post_hashtags(to="uid")
        self.posts_are_about = self._map_post_hashtags(to="post_id")

        return self

    def _normalize_nested_dictionary_values(
        self, dictionary: DefaultDict[str, DefaultDict[str, float]]
    ) -> DefaultDict[str, DefaultDict[str, float]]:

        """
        make all values in a 2-level dictionary be between 0 and 1

        Parameters
        ----------
        dictionary
            2-level dictionary

        Returns
        -------
        2-level dictionary with adjusted values
        """

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

        # are all post IDs valid?
        valid_post_ids = {
            pid for pid in self.posts["post_id"] if self._is_valid_id(pid)
        }
        print(
            f'valid post IDs: {100*len(valid_post_ids)/self.posts["post_id"].nunique():.2f}%'
        )

        return self

    def find_out_more_about_users(self) -> "CatFeeder":

        """
        find out users age and gender; also an age group they fall into
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

    def get_tag_similarity_score(
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

    def get_post_similarity_scores(
        self,
        parts_of_speech: List[Literal["NOUN", "VERB"]],
        text_weight: float = 0.80,
        hashtag_weight: float = 0.20,
        family_boost: float = 1.50,
        family_min_similarity_score: float = 0.5,
    ) -> "CatFeeder":
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

        if sum([text_weight, hashtag_weight]) != 1:
            raise ValueError(f"text and hashtag weights must add up to 1")

        self.posts["post_text_processed"] = self.posts["text"].apply(
            lambda txt: {
                w.lemma_ for w in nlp(txt.lower()) if w.pos_ in parts_of_speech
            }
        )

        self.post_text_similarity = self.get_tag_similarity_score(
            self.posts[["post_id", "post_text_processed"]]
            .groupby("post_id")
            .agg(lambda x: set(chain.from_iterable(x)))
            .to_dict()["post_text_processed"],
            normalize=True,
        )

        self.posts_are_about = self._map_post_hashtags(to="post_id")
        self.post_tag_similarity = self.get_tag_similarity_score(
            self.posts_are_about, normalize=True
        )

        self.post_similarity = defaultdict(lambda: defaultdict(float))

        for pid1 in self.post_text_similarity:
            for pid2 in self.post_tag_similarity[pid1]:
                self.post_similarity[pid1][pid2] = (
                    self.post_text_similarity[pid1][pid2] * text_weight
                    + self.post_tag_similarity[pid1][pid2] * hashtag_weight
                )

        self.post_id_to_parent_post_id = {
            _[0]: _[1]
            for _ in self.posts[self.posts["parent_id"].apply(self._is_valid_id)][
                ["post_id", "parent_id"]
            ].to_dict(orient="split")["data"]
        }

        # boost similarity for post family members
        for pid in self.post_similarity:
            # if pid is someone's child
            if pid in self.post_id_to_parent_post_id:
                parent_pid = self.post_id_to_parent_post_id[pid]
                self.post_similarity[pid][parent_pid] = max(
                    family_min_similarity_score,
                    min(1, self.post_similarity[pid][parent_pid] * family_boost),
                )
                self.post_similarity[parent_pid][pid] = self.post_similarity[pid][
                    parent_pid
                ]

        return self

    def get_user_similarity_scores(
        self, demographics_weight: float = 0.5, interests_weight: float = 0.5
    ) -> "CatFeeder":

        """
        calculate similarity score for each two users

        Parameters
        ----------
        demographics_weight
            weight (importance) of demographic similarity
        interests_weight
            weight (importance) of interest similarity

        Returns
        -------
        self
        """

        if sum([demographics_weight, interests_weight]) != 1:
            raise ValueError(f"demographic and interests weights must add up to 1")

        self.interests = self.interests.groupby("uid").agg(set).to_dict()["interest"]

        self.user_interest_similarity = self.get_tag_similarity_score(
            self.interests,
            normalize=True,
        )

        self.users["age_and_gender"] = (
            (self.users["age_group"].astype(str) + " " + self.users["gender"])
            .str.split()
            .apply(set)
        )

        self.demographic_similarity = self.get_tag_similarity_score(
            self.users[["uid", "age_and_gender"]]
            .set_index("uid")
            .to_dict()["age_and_gender"],
            normalize=True,
        )

        self.user_similarity = defaultdict(lambda: defaultdict(float))

        for uid1 in self.demographic_similarity:
            for uid2 in self.demographic_similarity[uid1]:
                self.user_similarity[uid1][uid2] = (
                    self.demographic_similarity[uid1][uid2] * demographics_weight
                    + self.user_interest_similarity[uid1][uid2] * interests_weight
                )

        return self

    def feed(self, uid: str, current_time: datetime.datetime):

        if not self._is_valid_id(uid):
            raise ValueError(f"customer's ID {uid} is invalid!")

        if not self.users["uid"].str.contains(uid).any():
            raise Exception(f"sorry, we only serve customers from users.csv")

        print(f"nice to see you, {uid}")

        return self
