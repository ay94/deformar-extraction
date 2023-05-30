import pandas as pd
import plotly.express as px
class TokenAmbiguity:
    def __init__(self, subwords, tokens):
        self.subwords = subwords
        self.tokens = tokens
        token_tag_pairs = self.extract_token_tag_pair()
        self.visualize_ambiguity(token_tag_pairs)

    def extract_token_tag_pair(self):
        pairs = []
        if len(self.tokens) > 1:
            print()
            for token in self.tokens:
                for token_tag in self.subwords[token]:
                    pairs.append((token, token_tag['tag']))
        else:
            for token_tag in self.subwords[self.tokens[0]]:
                pairs.append((self.tokens[0], token_tag['tag']))
        return pairs

    def visualize_ambiguity(self, token_tag_pairs):
        # Create a dictionary of word-tag frequency counts
        word_tag_dict = {}
        for token, tag in token_tag_pairs:
            if token not in word_tag_dict:
                word_tag_dict[token] = {tag: 1}
            else:
                if tag not in word_tag_dict[token]:
                    word_tag_dict[token][tag] = 1
                else:
                    word_tag_dict[token][tag] += 1

        # Create a dataframe of word-tag frequency counts
        df = pd.DataFrame.from_dict(word_tag_dict, orient='index')

        df.fillna(value=0, inplace=True)
        # Create a heatmap using Plotly
        fig = px.imshow(df.T.values,
                        x=df.index,
                        y=df.columns,
                        color_continuous_scale='YlOrBr')

        fig.update_layout(
            title='Heatmap of Word Frequencies with All Tags',
            xaxis=dict(title='Words'),
            yaxis=dict(title='Tags'),
            height=500,
            width=800
        )

        fig.show()

