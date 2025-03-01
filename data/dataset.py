import torch
import torch_geometric as pyg
from torch_geometric.data import (
    HeteroData,
    InMemoryDataset
)

from tqdm import tqdm
import os
import shutil
import gc
import json
from datetime import datetime

import requests
import tarfile


class PHEME(InMemoryDataset):

    url = 'https://figshare.com/ndownloader/files/11767817'

    def __init__(
        self,
        root: str, tokenizer=None, embedder=None,
        device: torch.device=torch.device('cpu')
    ):

        self.tokenizer = tokenizer
        self.embedder  = embedder
        self.device    = device

        super().__init__(root)
        self.data, self.slices = torch.load(self.processed_paths[0])

    def raw_file_names(self):
        return ['PHEME']

    def processed_file_names(self):
        return ['PHEME.pt']

    def download(self):

        if not 'PHEME.bz2' in os.listdir(self.raw_dir):
            print('Downloading PHEME Dataset...')
            r = requests.get(self.url, allow_redirects=True)
            with open(os.path.join(self.raw_dir, 'PHEME.bz2'), 'wb') as f:
                f.write(r.content)

        if not 'PHEME' in os.listdir(self.raw_dir):
            print('Extracting...')

            with tarfile.open(os.path.join(self.raw_dir, 'PHEME.bz2'), 'r') as tar:
                tar.extractall(self.raw_dir)

            os.rename(os.path.join(self.raw_dir, 'all-rnr-annotated-threads'), os.path.join(self.raw_dir, 'PHEME'))

        if 'PHEME.json' not in os.listdir(
            os.path.join(self.raw_dir, 'PHEME')
        ):

            print('Aggregating all datas in one json file...')

            dataset = []

            for event in os.listdir(self.raw_paths[0]):

                if event.startswith('.'):
                    continue

                event_path = os.path.join(
                    self.raw_paths[0],
                    event
                )

                for rumourity in os.listdir(event_path):

                    if rumourity.startswith('.'):

                        continue

                    for news_id in tqdm(
                        os.listdir(
                            os.path.join(
                                event_path,
                                rumourity
                            )
                        ),
                        desc=f'{event[:-16]} --- {rumourity}'
                    ):

                        if news_id.startswith('.'):
                            continue

                        y = 1 if rumourity == 'rumours' else 0

                        news_path = os.path.join(
                            event_path,
                            rumourity,
                            news_id
                        )

                        tweet = json.load(
                            open(
                                os.path.join(
                                    news_path,
                                    'source-tweets',
                                    news_id + '.json'
                                ),
                                encoding='utf-8'
                            )
                        )

                        structure = json.load(
                            open(
                                os.path.join(
                                    news_path,
                                    'structure.json'
                                ),
                                encoding='utf-8'
                            )
                        )

                        reactions_path = os.path.join(
                            news_path,
                            'reactions'
                        )

                        reactions = []
                        if os.path.exists(reactions_path):

                            for filename in os.listdir(reactions_path):

                                if filename.startswith('.'):
                                    continue

                                reaction = json.load(
                                    open(
                                        os.path.join(
                                            reactions_path,
                                            filename
                                        ),
                                        encoding='utf-8'
                                    )
                                )
                                reactions.append(reaction)

                        dataset.append(
                            {
                                'tweet': tweet,
                                'reactions': reactions,
                                'structure': structure,
                                'event': event[:-16],
                                'label': y
                            }
                        )

            with open(
                os.path.join(
                    self.raw_dir,
                    'PHEME',
                    'PHEME.json'
                ),
                'w',
                encoding='utf-8'
            ) as f:

                json.dump(dataset, f)

    def process(self):

        # Load raw dataset
        conversations = json.load(
            open(
                os.path.join(
                    self.raw_dir,
                    'PHEME',
                    'PHEME.json'
                ),
                encoding='utf-8'
            )
        )

        dataset = []
        for conversation in tqdm(conversations, desc='Generating graph dataset...'):

            # extract important components
            root_tweet = conversation['tweet']
            reactions  = conversation['reactions']

            tweets = [root_tweet] + reactions
            users  = [tweet['user'] for tweet in tweets]
            # users out of conversation thread which
            # mentions from users in the conversation thread
            for tweet in tweets:

                mentions = tweet['entities']['user_mentions']
                users += mentions

            timestamp_idx            = self.__encode_timestamp(tweets=tweets, users=users)
            tweet_idx, unique_tweets = self.__id2index(tweets)
            user_idx, unique_users   = self.__id2index(users)

            # tweet-reply-tweet edges
            trt_edges, trt_time = self.__tweet__reply_tweet_edges(
                # only reaction tweet are replied to a tweet
                tweets=reactions,
                tweet_idx=tweet_idx,
                timestamp_idx=timestamp_idx
            )

            # user-write-tweet edges
            uwt_edges, uwt_time = self.__user_write_tweet_edges(
                tweets=tweets,
                user_idx=user_idx,
                tweet_idx=tweet_idx,
                timestamp_idx=timestamp_idx
            )

            # user-mention-user edges
            tmu_edges, tmu_time = self.__tweet_mention_user_edges(
                tweets=tweets,
                tweet_idx=tweet_idx,
                user_idx=user_idx,
                timestamp_idx=timestamp_idx
            )

            data = HeteroData()

            # data['tweet'].x = torch.ones((len(tweet_idx), 10))
            # data['user'].x = torch.ones((len(user_idx), 10))
            data['tweet'].x, data['user'].x = self.__encode_nodes(tweets=unique_tweets, users=unique_users)
            data['tweet', 'reply', 'tweet'].edge_index = torch.tensor(
                trt_edges
            ).t().long()
            data['tweet', 'reply', 'tweet'].time = torch.tensor(
                trt_time
            ).long()
            data['user', 'write', 'tweet'].edge_index = torch.tensor(
                uwt_edges
            ).t().long()
            data['user', 'write', 'tweet'].time = torch.tensor(
                uwt_time
            ).long()
            data['tweet', 'mention', 'user'].edge_index = torch.tensor(
                tmu_edges
            ).t().long()
            data['tweet', 'mention', 'user'].time = torch.tensor(
                tmu_time
            ).long()

            data.y         = 1 if conversation['label'] == "True" else 0
            data.event     = conversation['event']
            data.timesteps = len(timestamp_idx)

            dataset.append(data)

            # break
        # del conversations
        torch.save(
            self.collate(dataset),
            self.processed_paths[0]
        )

    def __id2index(self, A):

        ids = {}
        uniques = []
        idx = 0
        for a in A:

            if a['id'] in ids:
                continue

            ids[a['id']] = idx
            uniques.append(a)
            idx += 1

        return ids, uniques

    def __tweet__reply_tweet_edges(self, tweets, tweet_idx, timestamp_idx):

        edge_index = []
        time       = []

        for tweet in tweets:

            source_id = tweet['id']
            dest_id   = tweet['in_reply_to_status_id']
            if dest_id not in tweet_idx:
                continue

            edge_index.append(
                [
                    tweet_idx[source_id],
                    tweet_idx[dest_id]
                ]
            )
            time.append(
                timestamp_idx[
                    datetime.strptime(
                        tweet['created_at'],
                        '%a %b %d %H:%M:%S %z %Y'
                    )
                ]
            )

        return edge_index, time

    def __user_write_tweet_edges(self, tweets, user_idx, tweet_idx, timestamp_idx):

        edge_index = []
        time       = []

        for tweet in tweets:

            user = tweet['user']

            source_id = user['id']
            dest_id   = tweet['id']

            edge_index.append(
                [
                    user_idx[source_id],
                    tweet_idx[dest_id]
                ]
            )
            time.append(
                timestamp_idx[
                    datetime.strptime(
                        tweet['created_at'],
                        '%a %b %d %H:%M:%S %z %Y'
                    )
                ]
            )

        return edge_index, time

    def __tweet_mention_user_edges(self, tweets, tweet_idx, user_idx, timestamp_idx):

        edge_index = []
        time       = []

        for tweet in tweets:

            mentions  = tweet['entities']['user_mentions']
            source_id = tweet['id']

            for mention in mentions:

                dest_id = mention['id']

                edge_index.append(
                    [
                        tweet_idx[source_id],
                        user_idx[dest_id]
                    ]
                )
                time.append(
                    timestamp_idx[
                        datetime.strptime(
                            tweet['created_at'],
                            '%a %b %d %H:%M:%S %z %Y'
                        )
                    ]
                )

        return edge_index, time

    def __encode_timestamp(self, tweets, users):
        # no need to encode user timestamps
        # because they do not affect on relations
        timestamps = [
            datetime.strptime(
                tweet['created_at'],
                '%a %b %d %H:%M:%S %z %Y'
            ) for tweet in tweets
        ]

        timestamp_idx = {}
        idx = 0

        for timestamp in sorted(timestamps):

            if timestamp in timestamp_idx:
                continue

            timestamp_idx[timestamp] = idx
            idx += 1

        return timestamp_idx

    def __encode_nodes(self, tweets, users):

        tweet_texts = []
        tweet_features = []
        for tweet in tweets:
            tweet_texts.append(tweet['text'])
            tweet_features.append(
                [
                    tweet['retweet_count'],
                    tweet['retweeted']
                ]
            )
        user_descs = []
        user_features = []
        for user in users:
            if 'description' in user:
                if user['description']:
                    user_descs.append(user['description'])
                else:
                    user_descs.append('')
                user_features.append(
                    [
                        user['followers_count'],
                        user['friends_count'],
                        user['verified']
                    ]
                )
            else:
                user_descs.append('OUT OF CONVERSATION THREAD')
                user_features.append(
                    [
                        -1,
                        -1,
                        0
                    ]
                )

        # tokenize and embed tweet and reactions
        Inputs = self.tokenizer(
            tweet_texts + user_descs,
            padding=True,
            truncation=True,
            return_tensors='pt',
            add_special_tokens=True
        ).to(self.device)
        # TweetInputs = self.tokenizer(
        #     tweet_texts,
        #     padding=True,
        #     truncation=True,
        #     return_tensors='pt',
        #     add_special_tokens=True
        # ).to(self.device)
        # # tokenize and embed user descriptions
        # UserInputs = self.tokenizer(
        #     user_descs,
        #     padding=True,
        #     truncation=True,
        #     return_tensors='pt',
        #     add_special_tokens=True
        # ).to(self.device)

        # get embeddings
        with torch.no_grad():

            # TweetOutputs = self.embedder(**TweetInputs)
            # UserOutputs = self.embedder(**UserInputs)

            Outputs = self.embedder(**Inputs)

            # tweet_embeddings = TweetOutputs.pooler_output
            # user_embeddings = UserOutputs.pooler_output

            embeddings = Outputs.pooler_output

            tweet_embeddings = embeddings[:len(tweet_texts)]
            user_embeddings = embeddings[len(tweet_texts):]

            tweet_features = torch.concat(
                [
                    tweet_embeddings.cpu(),
                    torch.tensor(tweet_features)
                ],
                dim=1
            )
            user_features=torch.concat(
                [
                    user_embeddings.cpu(),
                    torch.tensor(user_features)
                ],
                dim=1
            )

            ##### Free The Memory Cache (GPU) #####
            gc.collect()
            torch.cuda.empty_cache()
            gc.collect()
            #######################################
        return tweet_features, user_features









class PersianTweet(InMemoryDataset):

    label_dict = {
        'false': 1,
        'half false': 1,
        'deceptive': 1,
        'true': 0,
        'half true': 0,
        'Almost true': 0,
        'False': 1,
        'Misleading': 1,
        'Outrageous': 1,
        'True': 0,
        'No data': 0
    }

    def __init__(
        self,
        root: str, tokenizer=None, embedder=None,
        device: torch.device=torch.device('cpu')
    ):

        self.tokenizer = tokenizer
        self.embedder  = embedder
        self.device    = device

        super().__init__(root)
        self.data, self.slices = torch.load(self.processed_paths[0])

    def raw_file_names(self):
        return ['PersianTweet']

    def processed_file_names(self):
        return ['PersianTweet.pt']

    def download(self):

        if not os.path.exists(self.raw_dir):
            os.makedirs(self.raw_dir)
        
        if not 'PersianTweet' in os.listdir(self.raw_dir):

            shutil.copytree("PersianTweet", os.path.join(self.raw_dir, "PersianTweet"))

        if 'PersianTweet.json' not in os.listdir(
            os.path.join(self.raw_dir, 'PersianTweet')
        ):

            print('Aggregating all datas in one json file...')

            dataset = []

            dataset_path = os.path.join(
                self.raw_dir,
                "PersianTweet"
            )


            for news_id in tqdm(
                os.listdir(dataset_path)
            ):
                tweet = json.load(
                    open(
                        os.path.join(
                            dataset_path,
                            news_id,
                            "source-tweets",
                            f"{news_id}.json"
                        ),
                        encoding="utf-8"
                    )
                )
                reactions = []
                reactions_path = os.path.join(
                    dataset_path,
                    news_id,
                    "reactions"
                )
                if os.path.exists(reactions_path):
                    for reaction_path in os.listdir(
                        reactions_path
                    ):
                        
                        reaction = json.load(
                            open(
                                os.path.join(
                                    reactions_path,
                                    reaction_path
                                ),
                                encoding="utf-8"
                            )
                        )
                        reactions.append(reaction)

                y = tweet['label']

                dataset.append(
                    {
                        'tweet': tweet,
                        'reactions': reactions,
                        'label': y
                    }
                )
                

            with open(
                os.path.join(dataset_path, "PersianTweet.json"),
                'w',
                encoding='utf-8'
            ) as f:

                json.dump(dataset, f)

    def process(self):

        # Load raw dataset
        conversations = json.load(
            open(
                os.path.join(
                    self.raw_dir,
                    "PersianTweet",
                    'PersianTweet.json'
                ),
                encoding='utf-8'
            )
        )

        dataset = []
        for conversation in tqdm(conversations, desc='Generating graph dataset...'):

            # extract important components
            root_tweet = conversation['tweet']
            reactions  = conversation['reactions']

            tweets = [root_tweet] + reactions
            users  = [tweet['author'] for tweet in tweets]
            # users out of conversation thread which
            # mentions from users in the conversation thread
            for tweet in tweets:

                mentions = tweet['entities']['user_mentions']
                for mention in mentions:
                    mention['id'] = mention['id_str']

                users += mentions
        
            timestamp_idx            = self.__encode_timestamp(tweets=tweets, users=users)
            tweet_idx, unique_tweets = self.__id2index(tweets)
            user_idx, unique_users   = self.__id2index(users)

            # tweet-reply-tweet edges
            trt_edges, trt_time = self.__tweet__reply_tweet_edges(
                # only reaction tweet are replied to a tweet
                tweets=reactions,
                tweet_idx=tweet_idx,
                timestamp_idx=timestamp_idx
            )

            # user-write-tweet edges
            uwt_edges, uwt_time = self.__user_write_tweet_edges(
                tweets=tweets,
                user_idx=user_idx,
                tweet_idx=tweet_idx,
                timestamp_idx=timestamp_idx
            )

            # user-mention-user edges
            tmu_edges, tmu_time = self.__tweet_mention_user_edges(
                tweets=tweets,
                tweet_idx=tweet_idx,
                user_idx=user_idx,
                timestamp_idx=timestamp_idx
            )

            data = HeteroData()

            # data['tweet'].x = torch.ones((len(tweet_idx), 10))
            # data['user'].x = torch.ones((len(user_idx), 10))
            data['tweet'].x, data['user'].x = self.__encode_nodes(tweets=unique_tweets, users=unique_users)
            data['tweet', 'reply', 'tweet'].edge_index = torch.tensor(
                trt_edges
            ).t().long()
            data['tweet', 'reply', 'tweet'].time = torch.tensor(
                trt_time
            ).long()
            data['user', 'write', 'tweet'].edge_index = torch.tensor(
                uwt_edges
            ).t().long()
            data['user', 'write', 'tweet'].time = torch.tensor(
                uwt_time
            ).long()
            data['tweet', 'mention', 'user'].edge_index = torch.tensor(
                tmu_edges
            ).t().long()
            data['tweet', 'mention', 'user'].time = torch.tensor(
                tmu_time
            ).long()

            data.y         = self.label_dict[conversation['label']]
            data.timesteps = len(timestamp_idx)

            dataset.append(data)

            # break
        # del conversations
        torch.save(
            self.collate(dataset),
            self.processed_paths[0]
        )

    def __id2index(self, A):

        ids = {}
        uniques = []
        idx = 0
        for a in A:

            if a['id'] in ids:
                continue

            ids[a['id']] = idx
            uniques.append(a)
            idx += 1

        return ids, uniques

    def __tweet__reply_tweet_edges(self, tweets, tweet_idx, timestamp_idx):

        edge_index = []
        time       = []

        for tweet in tweets:

            print(tweet)

            source_id = tweet['id']
            try:
                dest_id   = tweet['inReplyToId']
            except:
                dest_id = tweet['conversationId']
            if dest_id not in tweet_idx:
                continue

            edge_index.append(
                [
                    tweet_idx[source_id],
                    tweet_idx[dest_id]
                ]
            )
            time.append(
                timestamp_idx[
                    datetime.strptime(
                        tweet['createdAt'],
                        '%a %b %d %H:%M:%S %z %Y'
                    )
                ]
            )

        return edge_index, time

    def __user_write_tweet_edges(self, tweets, user_idx, tweet_idx, timestamp_idx):

        edge_index = []
        time       = []

        for tweet in tweets:

            user = tweet['author']

            source_id = user['id']
            dest_id   = tweet['id']

            edge_index.append(
                [
                    user_idx[source_id],
                    tweet_idx[dest_id]
                ]
            )
            time.append(
                timestamp_idx[
                    datetime.strptime(
                        tweet['createdAt'],
                        '%a %b %d %H:%M:%S %z %Y'
                    )
                ]
            )

        return edge_index, time

    def __tweet_mention_user_edges(self, tweets, tweet_idx, user_idx, timestamp_idx):

        edge_index = []
        time       = []

        for tweet in tweets:

            mentions  = tweet['entities']['user_mentions']
            source_id = tweet['id']

            for mention in mentions:

                dest_id = mention['id']

                edge_index.append(
                    [
                        tweet_idx[source_id],
                        user_idx[dest_id]
                    ]
                )
                time.append(
                    timestamp_idx[
                        datetime.strptime(
                            tweet['createdAt'],
                            '%a %b %d %H:%M:%S %z %Y'
                        )
                    ]
                )

        return edge_index, time

    def __encode_timestamp(self, tweets, users):
        # no need to encode user timestamps
        # because they do not affect on relations
        timestamps = [
            datetime.strptime(
                tweet['createdAt'],
                '%a %b %d %H:%M:%S %z %Y'
            ) for tweet in tweets
        ]

        timestamp_idx = {}
        idx = 0

        for timestamp in sorted(timestamps):

            if timestamp in timestamp_idx:
                continue

            timestamp_idx[timestamp] = idx
            idx += 1

        return timestamp_idx

    def __encode_nodes(self, tweets, users):

        tweet_texts = []
        tweet_features = []
        for tweet in tweets:
            tweet_texts.append(tweet['text'])
            tweet_features.append(
                [
                    tweet['retweetCount'],
                    tweet['isRetweet']
                ]
            )
        user_descs = []
        user_features = []
        for user in users:
            if 'description' in user:
                if user['description']:
                    user_descs.append(user['description'])
                else:
                    user_descs.append('')
                user_features.append(
                    [
                        user['followers'],
                        user['following'],
                        user['isVerified']
                    ]
                )
            else:
                user_descs.append('OUT OF CONVERSATION THREAD')
                user_features.append(
                    [
                        -1,
                        -1,
                        0
                    ]
                )

        # tokenize and embed tweet and reactions
        Inputs = self.tokenizer(
            tweet_texts + user_descs,
            padding=True,
            truncation=True,
            return_tensors='pt',
            add_special_tokens=True
        ).to(self.device)

        # get embeddings
        with torch.no_grad():


            Outputs = self.embedder(**Inputs)

            embeddings = Outputs.pooler_output

            tweet_embeddings = embeddings[:len(tweet_texts)]
            user_embeddings = embeddings[len(tweet_texts):]

            tweet_features = torch.concat(
                [
                    tweet_embeddings.cpu(),
                    torch.tensor(tweet_features)
                ],
                dim=1
            )
            user_features=torch.concat(
                [
                    user_embeddings.cpu(),
                    torch.tensor(user_features)
                ],
                dim=1
            )

            ##### Free The Memory Cache (GPU) #####
            gc.collect()
            torch.cuda.empty_cache()
            gc.collect()
            #######################################
        return tweet_features, user_features
