#!/usr/bin/env python3
"""S/D/I and length-ratio pre-flight check, §2b step 1a.

Ref/hyp pairs are copy-pasted verbatim from the final ([eval_dev_clean] and
[eval_dev_other] step 8652) sample-generation blocks in
logs/melt-train-container.45969297.out (MA-librispeech-l4-ep3, MN5).
Normalization mirrors melt/training/metrics.py's TrainingEvaluator exactly
(BasicTextNormalizer, Whisper-style: lowercase, strip bracketed/parenthetical
asides, strip symbols/punctuation via unicodedata categories, collapse
whitespace) so results are comparable to the trainer's own reported WER/CER.
"""
import re
import unicodedata
import jiwer


def remove_symbols(s: str) -> str:
    return "".join(
        " " if unicodedata.category(c)[0] in "MSP" else c
        for c in unicodedata.normalize("NFKC", s)
    )


class BasicTextNormalizer:
    def __call__(self, s: str) -> str:
        s = s.lower()
        s = re.sub(r"[<\[][^>\]]*[>\]]", "", s)
        s = re.sub(r"\(([^)]+?)\)", "", s)
        s = remove_symbols(s).lower()
        s = re.sub(r"\s+", " ", s)
        return s


norm = BasicTextNormalizer()

# (set, idx, ref, hyp)
pairs = [
("dev_clean", 0, "at this turning point of history, there manifest themselves side by side and often mixed and entangled together a magnificent manifold virgin forest like up growth and up striving, a kind of tropical tempo in the rivalry of growth and an extraordinary decay and self-destruction owing to the savagely opposing and seemingly exploding egoisms which strive with one another for sun and light and can no longer assign any limit, restraint, or forbearance for themselves by means of the hitherto existing morality.",
 "at this turning point of his journey there manifested himself man of fashion, whose style was so grand and lofty uproaring, opposing, and striking, with one another, forson and lyte, and canoel longe a siney in a linee, and canoel longe a siney in a linee...strive with one another forson and lyte, and canoel longe a siney in a linee...strive with one another forson and lyte, and canoel longe a siney in a linee...strive with one another forson and lyte...canoel longe a siney in a linee...strive with one another forson and lyte, and canoel longe a siney in a linee...strive with one another forson and lyte, and canoel longe a siney in a linee...strive with one another forson and lyte...canoel longe a siney in a linee...strive with one another forson and lyte, and canoel longe a siney in a linee...strive with one another forson and lyte, and canoel"),
("dev_clean", 1, "it was the season when the ancient sun god had been accustomed to receive his annual oblations, and we can well believe that those whose hearts still trembled at the name of bel must have connected the eclipse and the plague with the revolution in the national worship and the overthrow of the ancient gods on that plain of prostration where they had so long received the homage of an entire people.",
 "it was the season when the angels sung had been a custodian to the sea of his own ablation and we cannot well believe that those who heartstilled the natural wash of the ocean are gods on that plain of restoration where they had so long seen the omage of an eternity pale."),
("dev_clean", 2, "on perceiving this, claudia, when she had convinced herself that her beloved husband was no more, rent the air with her sighs and made the heavens ring with her lamentations. she tore her hair and scattered it to the winds; she beat her face with her hands and showed all the signs of grief and sorrow that could be conceived to come from an afflicted heart.",
 "on perceiving this closely, when she had convinced herself that her beloved husband had consented to the marriage, she tore her hair and gathered it to the winds, she became afraid with her hands as she stood all the sighs of grief and sorrow that could be conceived to come from a heart."),
("dev_clean", 3, "i was compelled by poverty to become a member of a musical band in which i could expect neither esteem nor consideration, and i was well aware that i should be the laughing stock of the persons who had known me as a doctor in divinity, as an ecclesiastic, and as an officer in the army, and had welcomed me in the highest society.",
 "i was compelled by authority to become a member of a miscellaneous band. in which i could expect neither assistance nor consideration; and as an actress, and an officer in the army, and had well known me in the highest society."),
("dev_clean", 4, "and if you have any desire to shorten the journey and put yourself easily in the way of salvation, come with me, and i will show you how to become a knight errant—a calling wherein so many hardships and mishaps are encountered that if they be taken as penances, they will lodge you in heaven in a trice.",
 "and if you have any desire to shoot the jerry and put yourself easily in the way of yourself, call with me, and i will show you how to be come in nice attire, a colling, wearing, and some many harps and miss hats are incounder that they be taken in as pen and sisters, they will lose you in a trip."),
("dev_clean", 5, "i felt that in my first profession, as i was not blessed with the vocation necessary to it, i should have succeeded only by dint of hypocrisy; and i should have been despicable in my own estimation even if i had seen the purple mantle on my shoulders. for the greatest dignities cannot silence a man's own conscience.",
 "i feel that in my first preference, as i was upblast with the vocation necessity to it, i should have been disappointed if it had been designed by my own estimate. even if i had seen the proper man at once, i should have seen the proper man at once."),
("dev_clean", 6, "ferns and palms, mosses and trees, and animals—all perfect, all beautiful—and yet all hidden away under this hill and turned into shining black coal. now i can very well remember when i first saw a coal fire and how odd it looked to see what seemed to be burning stones.",
 "four and pomegranates in trees and animals all perfumed, " + "all perfumed, " * 55 + "all perf"),
("dev_clean", 7, "surely i shall, if i give you and myself to the cause; and i do it gladly, though i know that my heart has got to ache as it never has ached yet when my courage fails as it will by and by, and my selfish soul counts the cost of my offering after the excitement is over.",
 "shall i shield if you and myself to the cause and do it gladly? though i know that my heart has got to ace as it never has a year, when my creatures fail as it will be by and by, and my self-soldiing soul comes the cause of my offering after the exception mine own are."),
("dev_clean", 8, "all watched with quickened breath and proud souls that living wave blue below and bright with a steely glitter above as it flowed down the street and away to join the sea of dauntless hearts that for months had rolled up against the south and ebbed back reddenied with the blood of men like these.",
 "all watch with quick breath and prudently souls that living wave of blue above as it flower down the street in a way to join the sea of don luscious that for a moment had roared up against the south and abated, rounded up again, the south, and departed, ranting with the bullet of men like these."),
("dev_clean", 9, "and i beheld the flamelets onward go, leaving behind themselves the air depicted; and they of trailing pennons had the semblance so that it overhead remained distinct with sevenfold lists, all of them of the colours whence the sun's bow is made and delia's girdle.",
 "and i believe held the flame on the left hand, on the right hand, " * 24 + "on the left hand, on the right hand,"),
("dev_other", 0, "on two opposite pages of the idiot one finds the following characters brought in by name: general epanchin, prince s. adelaida ivanovna, lizaveta prokofyevna, evgeny pavlovitch radomsky, princess belyokonsky, aglaya, prince myshkin, kolya ivolgin, ippolit, varya, ferdyshchenko, nastasya filippovna, nina alexandrovna, ganya ptitsyn, and general ivolgin.",
 "on two options of the edicts, one fines the following correctness by name. january 13, 1819, at aetna, i have given a version of the prophecy of edward the prophet, you philistine, who availed himself of the scriptures. a testament of philistine, neva, elijah, and jeremiah."),
("dev_other", 1, "we have now granted, therefore, thought not constituting the idea of god; and accordingly the idea of god does not naturally follow from its nature in so far as it is absolute thought. for it is conceived as constituting, and also as not constituting, the idea of god, which is against our hypothesis.",
 "we have now granted the force. thought, not considering the idea of god, and accompanying the idea of god, " + "which is a constant and always so as not considering the idea of god, " * 13 + "which is a constant and always so"),
("dev_other", 2, "i have written to the bishop; i dare say i have told you so, but i forget things. just now said mr. hale, collapsing into his depressed manner as soon as he came to talk of hard, matter-of-fact details informing him of my intention to resign this vicarage.",
 "i have rendered to the bishop i daresay i have told you so much that i must confess, i am afraid these things are now. and mister heller, calculating and trying to preserve manchester as soon as possible, he forming him one minute to resume the very nature of things..."),
("dev_other", 3, "niccolo da monte aguto to whom i had written wrote back saying that he had spoken to that mad melancholy philosopher lorenzino for it he had replied that he was thinking night and day of nothing else and that he would finish it as soon as he was able.",
 "nickel and ol' dudouk to home i had returned, robert bax, saying that he had spent two hours i had read, robert bax, saying that he had spent two hours i had read, robert bax, saying that he had spent two hours i had read."),
("dev_other", 4, "it fair made my hair rise to hear him with the big still solemn desert outside and the quiet moonlight and the shadows and him sitting up straight and gaunt, his eyes blazing each side his big eagle nose and his snaky hair hanging over the raw cut across his head.",
 "it farmed a hair-rise to hear in with a big still sound desiderous in the quiet moonlight and shadows and himsit up straight gone. his as basis in needy sat his big nose."),
("dev_other", 5, "he ceased, and tara starry-eyed thus to the angry prince replied: \"not to my lord! shouldst thou address a speech so fraught with bitterness, not thus reproached my lord should be; and least of all, o prince, by thee.\"",
 "he sees and towers afar, to the end of the ranges, to the end of the mountains. \"no,\" said so far to be nears; no, to the north, repitched to my lodish bed, upon little old priney by the door."),
("dev_other", 6, "as the children arrive at years of discretion, these treasures pass out of the mother's faithful keeping into the hands of their actual owners, and they are usually kept stored away in some little-used drawer or cabinet until death removes the necessity for any further safeguards over life.",
 "as the character of russia described, these traits past of the mother's face, the mother's face, the mother's face, the handsome aspect of her own, and her real character, storey and some little use, draw capton till death removes the succession from any further speech of a lifeface."),
("dev_other", 7, "the knowledge that such changes can occur flits over the mind like the thought of death, obscuring all our gay fancies with its bat-like wing and tainting the healthy atmosphere of our happiness with its venomous exhalations.",
 "the noise that such chains can utter is not such a cruel word. oh, gentle friends, be not afraid of him for he is a happy man. but the venemous egresses."),
("dev_other", 8, "raising her eyes in a slightly inquiring manner from the book she was reading, she leant back in the chair as if soaking herself in luxurious sensations which had nothing to do with him and replied to his greeting with a few commonplace words.",
 "rising high in a slightly increasing manner from the book she was reading, she led back into her chair as if soothing herself in luxuriant sentences which had nothing to do with him and repeated to his greeting with a few common phrases words."),
("dev_other", 9, "there are hints of the grand style of lyric poetry in these lines, but phrases like \"in your make\" and \"as the hope hour stroked its sum\" are discords that bring it tumbling to the levels of victorian commonplace.",
 "there's his house in the green stile of the little port in the next. before his lack of knowledge, end as the hubbub stirred it some, and disgust was the brain it tumbling to the lower places of being torn and combed."),
]

refs_n = [norm(r) for (_, _, r, h) in pairs]
hyps_n = [norm(h) for (_, _, r, h) in pairs]

overall = jiwer.process_words(refs_n, hyps_n)
print(f"n={len(pairs)} overall: WER={overall.wer:.3f} S={overall.substitutions} D={overall.deletions} I={overall.insertions} H={overall.hits}")
tot = overall.substitutions + overall.deletions + overall.insertions
print(f"  error split: S={overall.substitutions/tot:.1%} D={overall.deletions/tot:.1%} I={overall.insertions/tot:.1%}")

ref_words = sum(len(r.split()) for r in refs_n)
hyp_words = sum(len(h.split()) for h in hyps_n)
print(f"  length ratio (hyp/ref words): {hyp_words/ref_words:.3f}  (ref_words={ref_words} hyp_words={hyp_words})")

print()
print("per-example (normalized word counts, ref -> hyp, WER):")
runaway = []
for (name, idx, r, h), r_n, h_n in zip(pairs, refs_n, hyps_n):
    m = jiwer.process_words([r_n], [h_n])
    rw, hw = len(r_n.split()), len(h_n.split())
    flag = " <-- RUNAWAY/REPEAT" if hw > 2.5 * rw else ""
    if flag:
        runaway.append(f"{name}[{idx}]")
    print(f"  {name}[{idx}]: ref={rw:>3}w hyp={hw:>3}w  WER={m.wer:.2f}  S={m.substitutions} D={m.deletions} I={m.insertions}{flag}")

print()
print(f"Flagged as runaway/repeat (hyp > 2.5x ref length): {runaway}")

print()
print("--- excluding flagged runaway examples ---")
keep = [i for i, (name, idx, r, h) in enumerate(pairs) if f"{name}[{idx}]" not in runaway]
refs_k = [refs_n[i] for i in keep]
hyps_k = [hyps_n[i] for i in keep]
clean = jiwer.process_words(refs_k, hyps_k)
tot_c = clean.substitutions + clean.deletions + clean.insertions
print(f"n={len(keep)} WER={clean.wer:.3f} S={clean.substitutions} D={clean.deletions} I={clean.insertions}")
print(f"  error split: S={clean.substitutions/tot_c:.1%} D={clean.deletions/tot_c:.1%} I={clean.insertions/tot_c:.1%}")
rw_k = sum(len(r.split()) for r in refs_k)
hw_k = sum(len(h.split()) for h in hyps_k)
print(f"  length ratio: {hw_k/rw_k:.3f}")
