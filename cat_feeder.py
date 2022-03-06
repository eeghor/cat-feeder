import pandas as pd  # type: ignore
import re
from typing import List, Any, Dict, DefaultDict, Literal, Set, Union
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
    SECONDS_IN_MINUTE = 60
    HYPHENS_IN_ID = 4

    def __init__(
        self,
        user_details_file: str = "data/users.csv",
        user_interests_file: str = "data/interest[1].csv",
        posts_file: str = "data/posts.csv",
        min_user_similarity_score: float = 0.5,
        min_post_similarity_score: float = 0.1,
        max_posts_to_show: int = 3,
    ) -> None:

        self.user_details_file = user_details_file
        self.user_interests_file = user_interests_file
        self.posts_file = posts_file
        self.min_user_similarity_score = min_user_similarity_score
        self.min_post_similarity_score = min_post_similarity_score
        self.max_posts_to_show = max_posts_to_show
        self.users = pd.read_csv(self.user_details_file, parse_dates=["dob"])
        self.interests = pd.read_csv(self.user_interests_file)
        self.posts = pd.read_csv(self.posts_file, parse_dates=["post_time"])

    def _is_valid_id(self, some_id: Any) -> bool:

        """
        check if the supplied argument looks like a legit ID

        Parameters
        ----------
        some_id
          an argument that could be a valid ID

        Returns
        -------
        True if valid, False otherwise

        """
        return len(str(some_id).split("-")) == (self.HYPHENS_IN_ID + 1)

    def _map_post_hashtags(self, to: Literal["uid", "post_id"]) -> Dict[str, Set[str]]:

        """
        map post hashtags to either user IDs or post IDs

        Parameters
        ----------
        to
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

    def _normalize_dictionary_values(
        self, dictionary: Union[DefaultDict[str, float], Dict[str, float]]
    ) -> Dict[str, float]:

        """
        make values in a dictionary add up to 1
        """

        sum_all_values = sum(dictionary.values())

        return {key_: value_ / sum_all_values for key_, value_ in dictionary.items()}

    def _find_out_more_about_users(self) -> "CatFeeder":

        """
        find out users age and gender; also an age group they fall into
        """

        self.users["gender"] = (
            self.users["first_name"] + " " + self.users["family_name"]
        ).apply(lambda x: gd.get_gender(x).gender)
        self.users["age_years"] = (
            self.users["dob"]
            .apply(lambda x: (self.current_time - x).days / self.DAYS_IN_A_YEAR)
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

    def _get_tag_similarity_score(
        self,
        set_of_tags: Union[Dict[str, Set[str]], DefaultDict[str, Set[str]]],
        normalize: bool = False,
    ) -> DefaultDict[str, DefaultDict[str, float]]:

        """
        calculate similarity between sets of tags associated with certain IDs

        Parameters
        ----------
        set_of_tags
          is a dictionary of the form {ID: {tag1, tag2, ..}}
        normalize
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
            id1_tags = set_of_tags.get(id1, set())
            id2_tags = set_of_tags.get(id2, set())
            tag_similarities[id1][id2] = len(id1_tags & id2_tags)
            tag_similarities[id2][id1] = tag_similarities[id1][id2]

        if normalize:
            tag_similarities = self._normalize_nested_dictionary_values(
                tag_similarities
            )

        if not set(set_of_tags) == set(tag_similarities):
            print("some IDs are missing from simmilarity scores!")

        return tag_similarities

    def _get_post_similarity_scores(
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
        text_weight
          importance of text for similarity
        hashtag_weight
          importance of hashtags for similarity
        family_boost
          multiply similarity between two posts by this coefficient if they are family
        family_min_similarity_score
          similarity between two posts that are family can't be lower than this

        """

        if sum([text_weight, hashtag_weight]) != 1:
            raise ValueError(f"text and hashtag weights must add up to 1")

        self.posts["post_text_processed"] = self.posts["text"].apply(
            lambda txt: {
                w.lemma_ for w in nlp(txt.lower()) if w.pos_ in parts_of_speech
            }
        )

        self.post_text_similarity = self._get_tag_similarity_score(
            self.posts[["post_id", "post_text_processed"]]
            .groupby("post_id")
            .agg(lambda x: set(chain.from_iterable(x)))
            .to_dict()["post_text_processed"],
            normalize=True,
        )

        self.posts_are_about = self._map_post_hashtags(to="post_id")
        self.post_tag_similarity = self._get_tag_similarity_score(
            self.posts_are_about, normalize=True
        )

        self.post_similarity: DefaultDict[str, DefaultDict[str, float]] = defaultdict(
            lambda: defaultdict(float)
        )

        for pid1 in self.post_text_similarity:
            for pid2 in self.post_tag_similarity[pid1]:
                self.post_similarity[pid1][pid2] = (
                    self.post_text_similarity[pid1][pid2] * text_weight
                    + self.post_tag_similarity[pid1][pid2] * hashtag_weight
                )
                self.post_similarity[pid2][pid1] = self.post_similarity[pid1][pid2]

        self.post_id_to_parent_post_id = {
            _[0]: _[1]
            for _ in self.posts[self.posts["parent_id"].apply(self._is_valid_id)][
                ["post_id", "parent_id"]
            ].to_dict(orient="split")["data"]
        }

        self.post_replies_count = (
            self.posts[self.posts["parent_id"].apply(self._is_valid_id)][
                ["post_id", "parent_id"]
            ]
            .groupby("parent_id")
            .count()
            .to_dict()["post_id"]
        )

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

    def _get_user_similarity_scores(
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
        """

        if sum([demographics_weight, interests_weight]) != 1:
            raise ValueError(f"demographic and interests weights must add up to 1")

        self.interests = self.interests.groupby("uid").agg(set).to_dict()["interest"]

        self.user_interest_similarity = self._get_tag_similarity_score(
            self.interests,
            normalize=True,
        )

        self.users["age_and_gender"] = (
            (self.users["age_group"].astype(str) + " " + self.users["gender"])
            .str.split()
            .apply(set)
        )

        self.demographic_similarity = self._get_tag_similarity_score(
            self.users[["uid", "age_and_gender"]]
            .set_index("uid")
            .to_dict()["age_and_gender"],
            normalize=True,
        )

        self.user_similarity: DefaultDict[str, DefaultDict[str, float]] = defaultdict(
            lambda: defaultdict(float)
        )

        for uid1 in self.demographic_similarity:
            for uid2 in self.demographic_similarity[uid1]:
                self.user_similarity[uid1][uid2] = (
                    self.demographic_similarity[uid1][uid2] * demographics_weight
                    + self.user_interest_similarity[uid1][uid2] * interests_weight
                )

        return self

    def review_data(self) -> "CatFeeder":

        """
        review data to see if there's anything worth knowing;
        print findings

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

    def feed(self, uid: str, current_time: datetime.datetime) -> "CatFeeder":
        """
        decide which posts to feed to a user

        Parameters
        ----------
        uid
          an ID of this user who's supposed to be fed
        current_time
          timestamp when the user logs in (according to assignment rules, must be an argument)
        """

        if not self._is_valid_id(uid):
            raise ValueError(f"customer's ID {uid} is invalid!")

        if not self.users["uid"].str.contains(uid).any():
            print(f"sorry, we only serve customers from users.csv. no feed for you")
            return self

        self.current_time = current_time

        user_name = self.users.loc[self.users["uid"] == uid, "first_name"].squeeze()

        print(f"nice to see you, {user_name} :)")
        print(
            f"especially now when it's {self.current_time.strftime('%H:%M:%S %m/%d/%Y')}"
        )

        self._find_out_more_about_users()
        self._get_user_similarity_scores()
        self._get_post_similarity_scores(parts_of_speech=["NOUN"])

        self.posts_to_show = []

        # has this user posted anything?
        this_users_posts_latest_to_oldest = self.posts[
            self.posts["uid"] == uid
        ].sort_values("post_time", ascending=False)
        print(f"{user_name} has {len(this_users_posts_latest_to_oldest):,} post(s)")

        similar_users = self.user_similarity[uid]

        similar_other_users_most_to_least = sorted(
            [
                (uid, similarity_score)
                for uid_, similarity_score in similar_users.items()
                if (similarity_score >= self.min_user_similarity_score)
                and (uid_ != uid)
            ],
            key=lambda x: x[1],
            reverse=True,
        )

        posts_from_similar_users = []

        # take a user similar to this one and grab his latest post
        for similar_user_id, _ in similar_other_users_most_to_least:
            similar_users_posts = self.posts[self.posts["uid"] == similar_user_id]
            if not similar_users_posts.empty:
                posts_from_similar_users.append(
                    similar_users_posts.sort_values("post_time", ascending=False).iloc[
                        0
                    ]["post_id"]
                )
                if len(posts_from_similar_users) == self.max_posts_to_show:
                    break

        # this user has no posts
        if this_users_posts_latest_to_oldest.empty:
            self.posts_to_show = posts_from_similar_users

        # this user does have some posts
        else:
            for pid in set(this_users_posts_latest_to_oldest["post_id"]):

                # all posts similar to this one
                similar_posts = self.post_similarity.get(pid)

                if similar_posts:

                    similar_other_user_posts_most_to_least = sorted(
                        [
                            (pid_, similarity_score)
                            for pid_, similarity_score in similar_posts.items()
                            if (similarity_score > self.min_post_similarity_score)
                            and (pid_ != pid)
                        ],
                        key=lambda x: x[1],
                        reverse=True,
                    )

                    self.similar_other_user_posts_posted_ago = dict(
                        self.posts[
                            self.posts["post_id"].isin(
                                {_[0] for _ in similar_other_user_posts_most_to_least}
                            )
                        ]
                        .assign(
                            posted_minutes_ago=lambda x: (
                                current_time - x["post_time"]
                            ).dt.seconds
                            / self.SECONDS_IN_MINUTE
                        )[["post_id", "posted_minutes_ago"]]
                        .to_dict(orient="split")["data"]
                    )

                    self.post_replies_count_normalised = (
                        self._normalize_dictionary_values(self.post_replies_count)
                    )

                    # adjust similarity scores according to time passed since publication
                    # and number of replies
                    self.posts_to_show = [
                        pid
                        for pid, _ in sorted(
                            [
                                (
                                    pid_,
                                    (
                                        1
                                        + self.post_replies_count_normalised.get(
                                            pid_, 0
                                        )
                                    )
                                    * similarity_score
                                    / self.similar_other_user_posts_posted_ago[pid_],
                                )
                                for pid_, similarity_score in similar_other_user_posts_most_to_least
                            ],
                            key=lambda x: x[1],
                            reverse=True,
                        )
                    ][: self.max_posts_to_show]

            # no posts relevant (similar) enough? show posts from similar users then
            if not self.posts_to_show:
                self.posts_to_show = posts_from_similar_users

        return self
