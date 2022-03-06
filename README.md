![](pictures/cat-feeder.png)

#### A smart feed guided by semantic AI. You won't miss what matters.  

How does it work? In short, `Cat Feeder` runs an efficient proprietory algorithm on a vast number of data points to figure out similar users and similar posts. 

## Similar Users
To determine to what extent any two users are similar, our algorithm makes use of the following key data points:

* age group (as per user profile)
* gender (inferred by AI)
* interests (self-reported by user)

## Similar Posts
To establish similarity between any two posts our system focuses on 
* post text, i.e. what has been written
* hashtags added by users (if any)
* explicitly created relationships between posts, e.g. some are replies to others


This similarity knowledge is then used to understand what a user who just logged in may be into. One that's done, `Cat Feeder` reviews millions of posts to pick a few that deserve attention and shows them in the feed.



