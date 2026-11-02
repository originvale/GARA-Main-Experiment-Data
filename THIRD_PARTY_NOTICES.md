# Benchmark and passage attribution

The experimental records select and organize material from the following
benchmark training splits. The source question IDs, corpus identifiers, passage
titles, and source URLs are retained in the data files.

## 2WikiMultiHopQA

Xanh Ho, Anh-Khoa Duong Nguyen, Saku Sugawara, and Akiko Aizawa.
*Constructing A Multi-hop QA Dataset for Comprehensive Evaluation of Reasoning
Steps*. COLING 2020.

Source: https://github.com/Alab-NII/2wikimultihop

The source repository supplies an Apache-2.0 LICENSE:
https://github.com/Alab-NII/2wikimultihop/blob/main/LICENSE
Underlying Wikipedia and Wikidata material retains its source attribution and
terms; a repository's code license does not replace those terms.

## HotpotQA

Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio, William W. Cohen, Ruslan
Salakhutdinov, and Christopher D. Manning.
*HotpotQA: A Dataset for Diverse, Explainable Multi-hop Question Answering*.
EMNLP 2018.

Source: https://hotpotqa.github.io/

The project distributes its benchmark and processed Wikipedia corpus under
CC BY-SA 4.0: https://creativecommons.org/licenses/by-sa/4.0/.

## MuSiQue

Harsh Trivedi, Niranjan Balasubramanian, Tushar Khot, and Ashish Sabharwal.
*MuSiQue: Multihop Questions via Single-hop Question Composition*.
Transactions of the Association for Computational Linguistics, 2022.

Source: https://github.com/StonyBrookNLP/musique

The project states that MuSiQue is distributed under CC BY 4.0:
https://creativecommons.org/licenses/by/4.0/.

## Wikipedia passages

Retrieved Wikipedia passages are attributed to their contributing authors
through the retained article title and source URL. The original passage text is
stored as returned by the benchmark-specific retrieval corpus. Article author
history is available from the linked Wikipedia page. Applicable source terms,
including CC BY-SA, continue to apply to those passages.

The release adds experiment outputs, evaluation scores, and allocation records;
selection and serialization into CSV/JSON Lines do not change the source text.
