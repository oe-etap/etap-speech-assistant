#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Citation registry for the response evaluation toolkit.

Every metric implemented in this package names the publication it is derived
from. Keeping the mapping in one place means a report can list exactly which
literature backs the numbers it shows, and a paper can import the matching
BibTeX entries without transcription errors.

The `status` field records how strong the claim of validation actually is,
because the three cases must not be reported as if they were equivalent:

    "verifiable"  The check is decidable by construction. Nothing is estimated,
                  so there is no measurement error to validate away.
    "validated"   A psychometrically or empirically validated instrument, or a
                  statistic with published sampling theory.
    "established" A widely used research method with published human-correlation
                  evidence, but no validated instrument status. Requires local
                  calibration against human ratings before it carries weight.
    "surrogate"   A deliberate simplification of a published method, adopted to
                  avoid a heavy dependency. Reports the shape of the original
                  metric, not its published performance.
"""

from typing import Dict, Iterable, List, NamedTuple


class Reference(NamedTuple):
    """A single literature entry backing one or more metrics."""

    key: str            # BibTeX citation key
    short: str          # Short form for console output, e.g. "Zhou et al., 2023"
    title: str
    venue: str
    status: str         # One of: verifiable, validated, established, surrogate
    bibtex: str


def _ref(key, short, title, venue, status, bibtex):
    return Reference(key=key, short=short, title=title, venue=venue,
                     status=status, bibtex=bibtex.strip())


REFERENCES: Dict[str, Reference] = {

    # ---------------------------------------------------------------- Tier 0
    "ifeval": _ref(
        "zhou2023ifeval", "Zhou et al., 2023",
        "Instruction-Following Evaluation for Large Language Models",
        "arXiv:2311.07911", "verifiable",
        """
@misc{zhou2023ifeval,
  title={Instruction-Following Evaluation for Large Language Models},
  author={Zhou, Jeffrey and Lu, Tianjian and Mishra, Swaroop and Brahma, Siddhartha
          and Basu, Sujoy and Luan, Yi and Zhou, Denny and Hou, Le},
  year={2023},
  note={arXiv:2311.07911}
}
"""),

    "followbench": _ref(
        "jiang2024followbench", "Jiang et al., 2024",
        "FollowBench: A Multi-level Fine-grained Constraints Following Benchmark",
        "ACL 2024", "established",
        """
@inproceedings{jiang2024followbench,
  title={{FollowBench}: A Multi-level Fine-grained Constraints Following
         Benchmark for Large Language Models},
  author={Jiang, Yuxin and Wang, Yufei and Zeng, Xingshan and Zhong, Wanjun and
          Li, Liangyou and Mi, Fei and Shang, Lifeng and Jiang, Xin and
          Liu, Qun and Wang, Wei},
  booktitle={Proceedings of the 62nd Annual Meeting of the Association for
             Computational Linguistics},
  year={2024}
}
"""),

    "flesch1948": _ref(
        "flesch1948readability", "Flesch, 1948",
        "A new readability yardstick",
        "Journal of Applied Psychology 32(3)", "validated",
        """
@article{flesch1948readability,
  title={A new readability yardstick},
  author={Flesch, Rudolph},
  journal={Journal of Applied Psychology},
  volume={32},
  number={3},
  pages={221--233},
  year={1948},
  doi={10.1037/h0057532}
}
"""),

    "kincaid1975": _ref(
        "kincaid1975derivation", "Kincaid et al., 1975",
        "Derivation of new readability formulas for Navy enlisted personnel",
        "Naval Technical Training Command, Research Branch Report 8-75",
        "validated",
        """
@techreport{kincaid1975derivation,
  title={Derivation of New Readability Formulas (Automated Readability Index,
         Fog Count and Flesch Reading Ease Formula) for Navy Enlisted Personnel},
  author={Kincaid, J. Peter and Fishburne, Robert P. and Rogers, Richard L. and
          Chissom, Brad S.},
  year={1975},
  number={Research Branch Report 8-75},
  institution={Naval Technical Training Command, Millington TN}
}
"""),

    # ---------------------------------------------------------------- Tier 1
    "squad_f1": _ref(
        "rajpurkar2016squad", "Rajpurkar et al., 2016",
        "SQuAD: 100,000+ Questions for Machine Comprehension of Text "
        "(token-level F1)",
        "EMNLP 2016", "established",
        """
@inproceedings{rajpurkar2016squad,
  title={{SQuAD}: 100,000+ Questions for Machine Comprehension of Text},
  author={Rajpurkar, Pranav and Zhang, Jian and Lopyrev, Konstantin and
          Liang, Percy},
  booktitle={Proceedings of the 2016 Conference on Empirical Methods in
             Natural Language Processing},
  pages={2383--2392},
  year={2016},
  doi={10.18653/v1/D16-1264}
}
"""),

    "squad2": _ref(
        "rajpurkar2018know", "Rajpurkar et al., 2018",
        "Know What You Don't Know: Unanswerable Questions for SQuAD "
        "(answerable and unanswerable items)",
        "ACL 2018", "established",
        """
@inproceedings{rajpurkar2018know,
  title={Know What You Don't Know: Unanswerable Questions for {SQuAD}},
  author={Rajpurkar, Pranav and Jia, Robin and Liang, Percy},
  booktitle={Proceedings of the 56th Annual Meeting of the Association for
             Computational Linguistics},
  pages={784--789},
  year={2018},
  doi={10.18653/v1/P18-2124}
}
"""),

    "answer_presence": _ref(
        "chen2017reading", "Chen et al., 2017",
        "Reading Wikipedia to Answer Open-Domain Questions "
        "(answer presence in the generated text)",
        "ACL 2017", "established",
        """
@inproceedings{chen2017reading,
  title={Reading {W}ikipedia to Answer Open-Domain Questions},
  author={Chen, Danqi and Fisch, Adam and Weston, Jason and Bordes, Antoine},
  booktitle={Proceedings of the 55th Annual Meeting of the Association for
             Computational Linguistics},
  pages={1870--1879},
  year={2017},
  doi={10.18653/v1/P17-1171}
}
"""),

    "rouge": _ref(
        "lin2004rouge", "Lin, 2004",
        "ROUGE: A Package for Automatic Evaluation of Summaries",
        "ACL Workshop on Text Summarization 2004", "established",
        """
@inproceedings{lin2004rouge,
  title={{ROUGE}: A Package for Automatic Evaluation of Summaries},
  author={Lin, Chin-Yew},
  booktitle={Text Summarization Branches Out: Proceedings of the ACL-04 Workshop},
  pages={74--81},
  year={2004}
}
"""),

    "bertscore": _ref(
        "zhang2020bertscore", "Zhang et al., 2020",
        "BERTScore: Evaluating Text Generation with BERT",
        "ICLR 2020", "established",
        """
@inproceedings{zhang2020bertscore,
  title={{BERTScore}: Evaluating Text Generation with {BERT}},
  author={Zhang, Tianyi and Kishore, Varsha and Wu, Felix and
          Weinberger, Kilian Q. and Artzi, Yoav},
  booktitle={International Conference on Learning Representations},
  year={2020}
}
"""),

    "howntoeval": _ref(
        "liu2016hownotto", "Liu et al., 2016",
        "How NOT To Evaluate Your Dialogue System: An Empirical Study of "
        "Unsupervised Evaluation Metrics for Dialogue Response Generation",
        "EMNLP 2016", "validated",
        """
@inproceedings{liu2016hownotto,
  title={How {NOT} To Evaluate Your Dialogue System: An Empirical Study of
         Unsupervised Evaluation Metrics for Dialogue Response Generation},
  author={Liu, Chia-Wei and Lowe, Ryan and Serban, Iulian and
          Noseworthy, Michael and Charlin, Laurent and Pineau, Joelle},
  booktitle={Proceedings of the 2016 Conference on Empirical Methods in
             Natural Language Processing},
  pages={2122--2132},
  year={2016},
  doi={10.18653/v1/D16-1230}
}
"""),

    # ---------------------------------------------------------------- Tier 2
    "selfcheckgpt": _ref(
        "manakul2023selfcheckgpt", "Manakul et al., 2023",
        "SelfCheckGPT: Zero-Resource Black-Box Hallucination Detection",
        "EMNLP 2023", "established",
        """
@inproceedings{manakul2023selfcheckgpt,
  title={{SelfCheckGPT}: Zero-Resource Black-Box Hallucination Detection for
         Generative Large Language Models},
  author={Manakul, Potsawee and Liusie, Adian and Gales, Mark J. F.},
  booktitle={Proceedings of the 2023 Conference on Empirical Methods in
             Natural Language Processing},
  pages={9004--9017},
  year={2023},
  doi={10.18653/v1/2023.emnlp-main.557}
}
"""),

    "factscore": _ref(
        "min2023factscore", "Min et al., 2023",
        "FActScore: Fine-grained Atomic Evaluation of Factual Precision in "
        "Long Form Text Generation",
        "EMNLP 2023", "established",
        """
@inproceedings{min2023factscore,
  title={{FActScore}: Fine-grained Atomic Evaluation of Factual Precision in
         Long Form Text Generation},
  author={Min, Sewon and Krishna, Kalpesh and Lyu, Xinxi and Lewis, Mike and
          Yih, Wen-tau and Koh, Pang Wei and Iyyer, Mohit and
          Zettlemoyer, Luke and Hajishirzi, Hannaneh},
  booktitle={Proceedings of the 2023 Conference on Empirical Methods in
             Natural Language Processing},
  pages={12076--12100},
  year={2023},
  doi={10.18653/v1/2023.emnlp-main.741}
}
"""),

    "hallucination_survey": _ref(
        "ji2023hallucination", "Ji et al., 2023",
        "Survey of Hallucination in Natural Language Generation "
        "(intrinsic/extrinsic taxonomy)",
        "ACM Computing Surveys 55(12)", "validated",
        """
@article{ji2023hallucination,
  title={Survey of Hallucination in Natural Language Generation},
  author={Ji, Ziwei and Lee, Nayeon and Frieske, Rita and Yu, Tiezheng and
          Su, Dan and Xu, Yan and Ishii, Etsuko and Bang, Ye Jin and
          Madotto, Andrea and Fung, Pascale},
  journal={ACM Computing Surveys},
  volume={55},
  number={12},
  pages={1--38},
  year={2023},
  doi={10.1145/3571730}
}
"""),

    # ---------------------------------------------------------------- Tier 3
    "geval": _ref(
        "liu2023geval", "Liu et al., 2023",
        "G-Eval: NLG Evaluation using GPT-4 with Better Human Alignment",
        "EMNLP 2023", "established",
        """
@inproceedings{liu2023geval,
  title={{G-Eval}: {NLG} Evaluation using {GPT-4} with Better Human Alignment},
  author={Liu, Yang and Iter, Dan and Xu, Yichong and Wang, Shuohang and
          Xu, Ruochen and Zhu, Chenguang},
  booktitle={Proceedings of the 2023 Conference on Empirical Methods in
             Natural Language Processing},
  pages={2511--2522},
  year={2023},
  doi={10.18653/v1/2023.emnlp-main.153}
}
"""),

    "mtbench": _ref(
        "zheng2023judging", "Zheng et al., 2023",
        "Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena "
        "(judge agreement, position and verbosity bias)",
        "NeurIPS 2023 Datasets and Benchmarks", "established",
        """
@inproceedings{zheng2023judging,
  title={Judging {LLM}-as-a-Judge with {MT-Bench} and {Chatbot Arena}},
  author={Zheng, Lianmin and Chiang, Wei-Lin and Sheng, Ying and
          Zhuang, Siyuan and Wu, Zhanghao and Zhuang, Yonghao and Lin, Zi and
          Li, Zhuohan and Li, Dacheng and Xing, Eric P. and Zhang, Hao and
          Gonzalez, Joseph E. and Stoica, Ion},
  booktitle={Advances in Neural Information Processing Systems 36:
             Datasets and Benchmarks Track},
  year={2023}
}
"""),

    "prometheus2": _ref(
        "kim2024prometheus2", "Kim et al., 2024",
        "Prometheus 2: An Open Source Language Model Specialized in "
        "Evaluating Other Language Models",
        "EMNLP 2024", "established",
        """
@inproceedings{kim2024prometheus2,
  title={{Prometheus 2}: An Open Source Language Model Specialized in
         Evaluating Other Language Models},
  author={Kim, Seungone and Suk, Juyoung and Longpre, Shayne and
          Lin, Bill Yuchen and Shin, Jamin and Welleck, Sean and
          Neubig, Graham and Lee, Moontae and Lee, Kyungjae and Seo, Minjoon},
  booktitle={Proceedings of the 2024 Conference on Empirical Methods in
             Natural Language Processing},
  year={2024}
}
"""),

    "poll": _ref(
        "verga2024juries", "Verga et al., 2024",
        "Replacing Judges with Juries: Evaluating LLM Generations with a "
        "Panel of Diverse Models",
        "arXiv:2404.18796", "established",
        """
@misc{verga2024juries,
  title={Replacing Judges with Juries: Evaluating {LLM} Generations with a
         Panel of Diverse Models},
  author={Verga, Pat and Hofstatter, Sebastian and Althammer, Sophia and
          Su, Yixuan and Piktus, Aleksandra and Arkhangorodsky, Arkady and
          Xu, Minjie and White, Naomi and Lewis, Patrick},
  year={2024},
  note={arXiv:2404.18796}
}
"""),

    # -------------------------------------------------- Domain-specific rubrics
    "quest": _ref(
        "tam2024quest", "Tam et al., 2024",
        "QUEST: A framework for human evaluation of large language models in "
        "healthcare derived from literature review",
        "npj Digital Medicine 7:258", "validated",
        """
@article{tam2024quest,
  title={A framework for human evaluation of large language models in
         healthcare derived from literature review},
  author={Tam, Thomas Yu Chow and Sivarajkumar, Sonish and Kapoor, Sumit and
          Stolyar, Alisa V. and Polanska, Katelyn and McCarthy, Karleigh R. and
          Osterhoudt, Hunter and Wu, Xizhi and Visweswaran, Shyam and
          Fu, Sunyang and Mathur, Piyush and Cacciamani, Giovanni E. and
          Sun, Cong and Peng, Yifan and Wang, Yanshan},
  journal={npj Digital Medicine},
  volume={7},
  pages={258},
  year={2024},
  doi={10.1038/s41746-024-01258-7}
}
"""),

    "medpalm": _ref(
        "singhal2023clinical", "Singhal et al., 2023",
        "Large language models encode clinical knowledge "
        "(multi-axis physician and lay rater rubric)",
        "Nature 620", "validated",
        """
@article{singhal2023clinical,
  title={Large language models encode clinical knowledge},
  author={Singhal, Karan and Azizi, Shekoofeh and Tu, Tao and
          Mahdavi, S. Sara and Wei, Jason and Chung, Hyung Won and
          Scales, Nathan and Tanwani, Ajay and Cole-Lewis, Heather and
          Pfohl, Stephen and others},
  journal={Nature},
  volume={620},
  number={7972},
  pages={172--180},
  year={2023},
  doi={10.1038/s41586-023-06291-1}
}
"""),

    "stade2024": _ref(
        "stade2024behavioral", "Stade et al., 2024",
        "Large language models could change the future of behavioral "
        "healthcare: a proposal for responsible development and evaluation",
        "npj Mental Health Research 3:12", "validated",
        """
@article{stade2024behavioral,
  title={Large language models could change the future of behavioral
         healthcare: a proposal for responsible development and evaluation},
  author={Stade, Elizabeth C. and Stirman, Shannon Wiltsey and
          Ungar, Lyle H. and Boland, Cody L. and Schwartz, H. Andrew and
          Yaden, David B. and Sedoc, Jo{\\~a}o and DeRubeis, Robert J. and
          Willer, Robb and Eichstaedt, Johannes C.},
  journal={npj Mental Health Research},
  volume={3},
  pages={12},
  year={2024},
  doi={10.1038/s44184-024-00056-z}
}
"""),

    "epitome": _ref(
        "sharma2020empathy", "Sharma et al., 2020",
        "EPITOME: A Computational Approach to Understanding Empathy "
        "Expressed in Text-Based Mental Health Support",
        "EMNLP 2020", "validated",
        """
@inproceedings{sharma2020empathy,
  title={A Computational Approach to Understanding Empathy Expressed in
         Text-Based Mental Health Support},
  author={Sharma, Ashish and Miner, Adam S. and Atkins, David C. and
          Althoff, Tim},
  booktitle={Proceedings of the 2020 Conference on Empirical Methods in
             Natural Language Processing},
  pages={5263--5276},
  year={2020},
  doi={10.18653/v1/2020.emnlp-main.425}
}
"""),

    "xstest": _ref(
        "roettger2024xstest", "Röttger et al., 2024",
        "XSTest: A Test Suite for Identifying Exaggerated Safety Behaviours "
        "in Large Language Models",
        "NAACL 2024", "established",
        """
@inproceedings{roettger2024xstest,
  title={{XSTest}: A Test Suite for Identifying Exaggerated Safety Behaviours
         in Large Language Models},
  author={R{\\"o}ttger, Paul and Kirk, Hannah Rose and Vidgen, Bertie and
          Attanasio, Giuseppe and Bianchi, Federico and Hovy, Dirk},
  booktitle={Proceedings of the 2024 Conference of the North American Chapter
             of the Association for Computational Linguistics},
  pages={5377--5400},
  year={2024}
}
"""),

    "donotanswer": _ref(
        "wang2024donotanswer", "Wang et al., 2024",
        "Do-Not-Answer: Evaluating Safeguards in LLMs",
        "EACL 2024 Findings", "established",
        """
@inproceedings{wang2024donotanswer,
  title={Do-Not-Answer: Evaluating Safeguards in {LLM}s},
  author={Wang, Yuxia and Li, Haonan and Han, Xudong and Nakov, Preslav and
          Baldwin, Timothy},
  booktitle={Findings of the Association for Computational Linguistics: EACL 2024},
  pages={896--911},
  year={2024}
}
"""),

    # ------------------------------------------------- Agreement and reliability
    "krippendorff": _ref(
        "krippendorff2018content", "Krippendorff, 2018",
        "Content Analysis: An Introduction to Its Methodology "
        "(alpha coefficient)",
        "Sage, 4th edition", "validated",
        """
@book{krippendorff2018content,
  title={Content Analysis: An Introduction to Its Methodology},
  author={Krippendorff, Klaus},
  edition={4},
  year={2018},
  publisher={Sage Publications}
}
"""),

    "cohen_kappa": _ref(
        "cohen1960kappa", "Cohen, 1960",
        "A coefficient of agreement for nominal scales",
        "Educational and Psychological Measurement 20(1)", "validated",
        """
@article{cohen1960kappa,
  title={A coefficient of agreement for nominal scales},
  author={Cohen, Jacob},
  journal={Educational and Psychological Measurement},
  volume={20},
  number={1},
  pages={37--46},
  year={1960},
  doi={10.1177/001316446002000104}
}
"""),

    "fleiss_kappa": _ref(
        "fleiss1971kappa", "Fleiss, 1971",
        "Measuring nominal scale agreement among many raters",
        "Psychological Bulletin 76(5)", "validated",
        """
@article{fleiss1971kappa,
  title={Measuring nominal scale agreement among many raters},
  author={Fleiss, Joseph L.},
  journal={Psychological Bulletin},
  volume={76},
  number={5},
  pages={378--382},
  year={1971},
  doi={10.1037/h0031619}
}
"""),

    "gwet_ac1": _ref(
        "gwet2008ac1", "Gwet, 2008",
        "Computing inter-rater reliability and its variance in the presence "
        "of high agreement (AC1)",
        "British Journal of Mathematical and Statistical Psychology 61(1)",
        "validated",
        """
@article{gwet2008ac1,
  title={Computing inter-rater reliability and its variance in the presence of
         high agreement},
  author={Gwet, Kilem Li},
  journal={British Journal of Mathematical and Statistical Psychology},
  volume={61},
  number={1},
  pages={29--48},
  year={2008},
  doi={10.1348/000711006X126600}
}
"""),

    "kappa_paradox": _ref(
        "feinstein1990paradox", "Feinstein and Cicchetti, 1990",
        "High agreement but low kappa: I. The problems of two paradoxes",
        "Journal of Clinical Epidemiology 43(6)", "validated",
        """
@article{feinstein1990paradox,
  title={High agreement but low kappa: I. The problems of two paradoxes},
  author={Feinstein, Alvan R. and Cicchetti, Domenic V.},
  journal={Journal of Clinical Epidemiology},
  volume={43},
  number={6},
  pages={543--549},
  year={1990},
  doi={10.1016/0895-4356(90)90158-L}
}
"""),

    "icc_shrout": _ref(
        "shrout1979icc", "Shrout and Fleiss, 1979",
        "Intraclass correlations: uses in assessing rater reliability",
        "Psychological Bulletin 86(2)", "validated",
        """
@article{shrout1979icc,
  title={Intraclass correlations: Uses in assessing rater reliability},
  author={Shrout, Patrick E. and Fleiss, Joseph L.},
  journal={Psychological Bulletin},
  volume={86},
  number={2},
  pages={420--428},
  year={1979},
  doi={10.1037/0033-2909.86.2.420}
}
"""),

    "icc_koo": _ref(
        "koo2016icc", "Koo and Li, 2016",
        "A Guideline of Selecting and Reporting Intraclass Correlation "
        "Coefficients for Reliability Research",
        "Journal of Chiropractic Medicine 15(2)", "validated",
        """
@article{koo2016icc,
  title={A Guideline of Selecting and Reporting Intraclass Correlation
         Coefficients for Reliability Research},
  author={Koo, Terry K. and Li, Mae Y.},
  journal={Journal of Chiropractic Medicine},
  volume={15},
  number={2},
  pages={155--163},
  year={2016},
  doi={10.1016/j.jcm.2016.02.012}
}
"""),

    "landis_koch": _ref(
        "landis1977measurement", "Landis and Koch, 1977",
        "The measurement of observer agreement for categorical data "
        "(benchmark scale)",
        "Biometrics 33(1)", "validated",
        """
@article{landis1977measurement,
  title={The measurement of observer agreement for categorical data},
  author={Landis, J. Richard and Koch, Gary G.},
  journal={Biometrics},
  volume={33},
  number={1},
  pages={159--174},
  year={1977},
  doi={10.2307/2529310}
}
"""),

    # ------------------------------------------------------------- Statistics
    "ppi": _ref(
        "angelopoulos2023ppi", "Angelopoulos et al., 2023",
        "Prediction-powered inference",
        "Science 382(6671)", "validated",
        """
@article{angelopoulos2023ppi,
  title={Prediction-powered inference},
  author={Angelopoulos, Anastasios N. and Bates, Stephen and
          Fannjiang, Clara and Jordan, Michael I. and Zrnic, Tijana},
  journal={Science},
  volume={382},
  number={6671},
  pages={669--674},
  year={2023},
  doi={10.1126/science.adi6000}
}
"""),

    "autoeval": _ref(
        "boyeau2024autoeval", "Boyeau et al., 2024",
        "AutoEval Done Right: Using Synthetic Data for Model Evaluation",
        "arXiv:2403.07008", "established",
        """
@misc{boyeau2024autoeval,
  title={{AutoEval} Done Right: Using Synthetic Data for Model Evaluation},
  author={Boyeau, Pierre and Angelopoulos, Anastasios N. and Yosef, Nir and
          Malik, Jitendra and Jordan, Michael I.},
  year={2024},
  note={arXiv:2403.07008}
}
"""),

    "bootstrap": _ref(
        "efron1993bootstrap", "Efron and Tibshirani, 1993",
        "An Introduction to the Bootstrap",
        "Chapman and Hall", "validated",
        """
@book{efron1993bootstrap,
  title={An Introduction to the Bootstrap},
  author={Efron, Bradley and Tibshirani, Robert J.},
  year={1993},
  publisher={Chapman and Hall},
  address={New York}
}
"""),

    "cliffs_delta": _ref(
        "cliff1993dominance", "Cliff, 1993",
        "Dominance statistics: Ordinal analyses to answer ordinal questions",
        "Psychological Bulletin 114(3)", "validated",
        """
@article{cliff1993dominance,
  title={Dominance statistics: Ordinal analyses to answer ordinal questions},
  author={Cliff, Norman},
  journal={Psychological Bulletin},
  volume={114},
  number={3},
  pages={494--509},
  year={1993},
  doi={10.1037/0033-2909.114.3.494}
}
"""),

    "tost": _ref(
        "lakens2017equivalence", "Lakens, 2017",
        "Equivalence Tests: A Practical Primer for t Tests, Correlations, "
        "and Meta-Analyses",
        "Social Psychological and Personality Science 8(4)", "validated",
        """
@article{lakens2017equivalence,
  title={Equivalence Tests: A Practical Primer for t Tests, Correlations,
         and Meta-Analyses},
  author={Lakens, Dani{\\"e}l},
  journal={Social Psychological and Personality Science},
  volume={8},
  number={4},
  pages={355--362},
  year={2017},
  doi={10.1177/1948550617697177}
}
"""),

    "holm": _ref(
        "holm1979simple", "Holm, 1979",
        "A simple sequentially rejective multiple test procedure",
        "Scandinavian Journal of Statistics 6(2)", "validated",
        """
@article{holm1979simple,
  title={A simple sequentially rejective multiple test procedure},
  author={Holm, Sture},
  journal={Scandinavian Journal of Statistics},
  volume={6},
  number={2},
  pages={65--70},
  year={1979}
}
"""),

    "wilcoxon": _ref(
        "wilcoxon1945individual", "Wilcoxon, 1945",
        "Individual comparisons by ranking methods (signed-rank test)",
        "Biometrics Bulletin 1(6)", "validated",
        """
@article{wilcoxon1945individual,
  title={Individual comparisons by ranking methods},
  author={Wilcoxon, Frank},
  journal={Biometrics Bulletin},
  volume={1},
  number={6},
  pages={80--83},
  year={1945},
  doi={10.2307/3001968}
}
"""),

    "mcnemar": _ref(
        "mcnemar1947note", "McNemar, 1947",
        "Note on the sampling error of the difference between correlated "
        "proportions or percentages",
        "Psychometrika 12(2)", "validated",
        """
@article{mcnemar1947note,
  title={Note on the sampling error of the difference between correlated
         proportions or percentages},
  author={McNemar, Quinn},
  journal={Psychometrika},
  volume={12},
  number={2},
  pages={153--157},
  year={1947},
  doi={10.1007/BF02295996}
}
"""),

    # ------------------------------------------- Reporting and spoken dialogue
    "tripod_llm": _ref(
        "gallifant2025tripodllm", "Gallifant et al., 2025",
        "The TRIPOD-LLM reporting guideline for studies using large "
        "language models",
        "Nature Medicine", "validated",
        """
@article{gallifant2025tripodllm,
  title={The {TRIPOD-LLM} reporting guideline for studies using large
         language models},
  author={Gallifant, Jack and Afshar, Majid and Ameen, Saleem and
          Aphinyanaphongs, Yindalon and Chen, Shan and Cacciamani, Giovanni and
          Demner-Fushman, Dina and Dligach, Dmitriy and Daneshjou, Roxana and
          Fernandes, Chrystinne and others},
  journal={Nature Medicine},
  volume={31},
  pages={60--69},
  year={2025},
  doi={10.1038/s41591-024-03425-5}
}
"""),

    "paradise": _ref(
        "walker1997paradise", "Walker et al., 1997",
        "PARADISE: A Framework for Evaluating Spoken Dialogue Agents",
        "ACL/EACL 1997", "validated",
        """
@inproceedings{walker1997paradise,
  title={{PARADISE}: A Framework for Evaluating Spoken Dialogue Agents},
  author={Walker, Marilyn A. and Litman, Diane J. and Kamm, Candace A. and
          Abella, Alicia},
  booktitle={Proceedings of the 35th Annual Meeting of the Association for
             Computational Linguistics},
  pages={271--280},
  year={1997},
  doi={10.3115/976909.979652}
}
"""),

    # ------------------------------- Recognizer fidelity and runtime behaviour
    "levenshtein": _ref(
        "levenshtein1966binary", "Levenshtein, 1966",
        "Binary codes capable of correcting deletions, insertions, and reversals",
        "Soviet Physics Doklady 10(8)", "verifiable",
        """
@article{levenshtein1966binary,
  title={Binary codes capable of correcting deletions, insertions, and
         reversals},
  author={Levenshtein, Vladimir I.},
  journal={Soviet Physics Doklady},
  volume={10},
  number={8},
  pages={707--710},
  year={1966}
}
"""),

    "wer_slu": _ref(
        "wang2003wer", "Wang et al., 2003",
        "Is word error rate a good indicator for spoken language understanding "
        "accuracy?",
        "IEEE ASRU 2003", "established",
        """
@inproceedings{wang2003wer,
  title={Is word error rate a good indicator for spoken language understanding
         accuracy?},
  author={Wang, Ye-Yi and Acero, Alex and Chelba, Ciprian},
  booktitle={2003 IEEE Workshop on Automatic Speech Recognition and
             Understanding (ASRU)},
  pages={577--582},
  year={2003},
  doi={10.1109/ASRU.2003.1318504}
}
"""),

    "tail_at_scale": _ref(
        "dean2013tail", "Dean and Barroso, 2013",
        "The Tail at Scale",
        "Communications of the ACM 56(2)", "established",
        """
@article{dean2013tail,
  title={The Tail at Scale},
  author={Dean, Jeffrey and Barroso, Luiz Andr{\\'e}},
  journal={Communications of the ACM},
  volume={56},
  number={2},
  pages={74--80},
  year={2013},
  doi={10.1145/2408776.2408794}
}
"""),

    "sassi": _ref(
        "hone2000sassi", "Hone and Graham, 2000",
        "Towards a tool for the Subjective Assessment of Speech System "
        "Interfaces (SASSI)",
        "Natural Language Engineering 6(3-4)", "validated",
        """
@article{hone2000sassi,
  title={Towards a tool for the Subjective Assessment of Speech System
         Interfaces ({SASSI})},
  author={Hone, Kate S. and Graham, Robert},
  journal={Natural Language Engineering},
  volume={6},
  number={3-4},
  pages={287--303},
  year={2000},
  doi={10.1017/S1351324900002497}
}
"""),
}


def cite(key: str) -> str:
    """Return the short console form of a reference, e.g. "Zhou et al., 2023"."""
    ref = REFERENCES.get(key)
    return ref.short if ref else f"<unknown reference: {key}>"


def status_of(key: str) -> str:
    """Return the validation status recorded for a reference key."""
    ref = REFERENCES.get(key)
    return ref.status if ref else "unknown"


def resolve(keys: Iterable[str]) -> List[Reference]:
    """Return the known references for `keys`, de-duplicated and sorted by key."""
    seen = {}
    for key in keys:
        ref = REFERENCES.get(key)
        if ref is not None:
            seen[ref.key] = ref
    return [seen[k] for k in sorted(seen)]


def bibliography_lines(keys: Iterable[str]) -> List[str]:
    """Render a human-readable bibliography for the methods actually used."""
    lines = []
    for ref in resolve(keys):
        lines.append(f"  [{ref.status:<11}] {ref.short}. {ref.title}. {ref.venue}.")
    return lines


def bibtex(keys: Iterable[str]) -> str:
    """Render BibTeX entries for `keys`, ready to append to a .bib file."""
    entries = [ref.bibtex for ref in resolve(keys)]
    header = ("% BibTeX entries for the evaluation methods used in this run.\n"
              "% Generated by evaluate_responses.py; check before committing.\n")
    return header + "\n\n".join(entries) + "\n"
