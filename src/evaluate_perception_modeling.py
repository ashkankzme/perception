from mrf_dataset import MRFDataset
from misinfo_perception_t5 import MisinfoPerceptionT5


if __name__ == '__main__':

    model = MisinfoPerceptionT5(1000000, loadLocally=True,
                                localModelPath='/local2/ashkank/perception/models/1_initial/trained_model')
    testDataPoint = [{
        "inputFrames": [
            "Headline: Dissecting the Debate: What Trump, Biden Said About COVID-19, Vaccines, and Obamacare\nReader's Reaction: Readers would feel tentative, Readers would want to carefuly skim the article\nWriter's Intent: The author is implying that policies on healthcare are contentious, The author is implying that there are strong opinions on vaccines, The author is implying that Covid-19 is being politicized\nPerceived Label: real news\n",
            "Headline: Climate Anarchy Spreads Across Germany As Protesters Attempt To Disrupt, Block Airports\nReader's Reaction: Readers would feel shocked, Readers would want to find out if there are local protests\nWriter's Intent: The author is implying that left-wingers are causing chaos, The author is implying that liberals are unreasonable\nPerceived Label: misinformation\n",
            "Headline: New Study finds no evidence of a 'signal of human-caused climate change' from weather extremes \u00c3\u00a2\u00e2\u0082\u00ac\u00e2\u0080\u009c Published in journal Environmental Hazards\nReader's Reaction: Readers would feel doubtful, Readers would want to read on the source\nWriter's Intent: The author is implying that society is not informed, The author is implying that left-wingers are lying, The author is implying that temperature changes are cyclical\nPerceived Label: misinformation\n",
            "Headline: Climate deniers get twice the news coverage of pro-climate messages, study finds\nReader's Reaction: Readers would feel disgust, Readers would want to join a group\nWriter's Intent: The author is implying that opposition is more vocal than unity, The author is implying that media indulges controversy, The author is implying that right-wingers are more entertaining to watch\nPerceived Label: real news\n",
            "Headline: Climate Hysteria Has Killed Academic Freedom \u00c3\u00a2\u00e2\u0082\u00ac\u00e2\u0080\u009c Peter Ridd loses, we all lose\nReader's Reaction: Readers would feel distrusting, Readers would want to find something less opinionated\nWriter's Intent: The author is implying that people are too gullible, The author is implying that leftism has overtaken academia\nPerceived Label: misinformation\n",
            "Headline: Climate crisis may be a factor in tufted puffins die-off, study says\nReader's Reaction: sad, read about puffins\nWriter's Intent: society isn't doing enough to save the environment, government could take action on environmental preservation, human climate change is killing animals\nPerceived Label: real news\n",
            "Headline: Youth Action Gives Me Hope Amid Climate Crisis and Trump's Totalitarian Threat\nReader's Reaction: irreverant, ignore the article\nWriter's Intent: the next generation will tackle climate change, the government is currently against the people, liberalism is rising against conservatism\nPerceived Label: misinformation\n",
            "Headline: Nasty Nets Cheer Biden Calling Trump 'Climate Arsonist'\nReader's Reaction: distrustful, skip the article\nWriter's Intent: the administration in power is anti-environment, the government is operating selfishly, tree cutting is an environmental problem\nPerceived Label: misinformation\n"
        ],
        "header": "Worker ID: AFIK3VBMMX6G6\nAge: 3\nGender: 2\nEducation: 3\nRace: white\nMedia Diet: washingtonPost, npr, other, newYorkTimes\n",
        "query": "Headline: Frigid in Chicago Must Mean Global Warming\nReader's Reactions: ?\nWriter's Intent: ?\nPerceived Label: ?\n",
        # "prediction": "Reader's Reactions: \nWriter's Intent: society overreacts, environmentalists over-apply global warming, temperatures are not rising\nPerceived Label: misinformation\n"
        "prediction": ""
    }]

    testInput = MRFDataset.formatInput(testDataPoint)




    input_ids = model.tokenizer(testInput[0]['X'], return_tensors='pt').input_ids
    output_ids = model.model.generate(input_ids)
    output = model.tokenizer.decode(output_ids[0], skip_special_tokens=True)
    print(output)