# 
# 
from 装饰器.decorator_程序启动 import logit
from textblob import TextBlob


@logit
def sentiment_analyse(txt):
    print(TextBlob(txt).sentiment)


if __name__ == '__main__':
    txt = '''
Stopping by Woods on a Snowy Evening


Whose woods these are I think I know.

His house is in the village though;

He will not see me stopping here

To watch his woods fill up with snow.


My little horse must think it queer

To stop without a farmhouse near

Between the woods and frozen lake

The darkest evening of the year.


He gives his harness bells a shake

To ask if there is some mistake.

The only other sound’s the sweep

Of easy wind and downy flake.
    '''
    sentiment_analyse(txt)

