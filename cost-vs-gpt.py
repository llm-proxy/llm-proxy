import os
import time

from dotenv import load_dotenv

from proxyllm.config.internal_config import internal_config
from proxyllm.provider.anthropic.claude import ClaudeAdapter
from proxyllm.provider.openai.chatgpt import OpenAIAdapter
from proxyllm.proxyllm import LLMProxy
from proxyllm.utils.cost import calculate_estimated_max_cost
from proxyllm.utils.tokenizer import bpe_tokenize_encode, vertexai_encode

# TODO - work on Proficiency routing vs gpt
# TODO - save out put to csv


load_dotenv(".env.test")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
HUGGING_FACE_API_KEY = os.getenv("HUGGING_FACE_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
GOOGLE_PROJECT_ID = os.getenv("GOOGLE_PROJECT_ID")
GOOGLE_APPLICATION_CREDENTIALS = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")


model_info = {
    "gpt-4-0125-preview": {
        "tokenizer": bpe_tokenize_encode,
        "cost_per_token_input": internal_config[0]["models"][0]["cost_per_token_input"],
        "cost_per_token_output": internal_config[0]["models"][0][
            "cost_per_token_output"
        ],
    },
    "gpt-4-1106-preview": {
        "tokenizer": bpe_tokenize_encode,
        "cost_per_token_input": internal_config[0]["models"][1]["cost_per_token_input"],
        "cost_per_token_output": internal_config[0]["models"][1][
            "cost_per_token_output"
        ],
    },
    "gpt-4": {
        "tokenizer": bpe_tokenize_encode,
        "cost_per_token_input": internal_config[0]["models"][2]["cost_per_token_input"],
        "cost_per_token_output": internal_config[0]["models"][2][
            "cost_per_token_output"
        ],
    },
    "gpt-4-32k": {
        "tokenizer": bpe_tokenize_encode,
        "cost_per_token_input": internal_config[0]["models"][3]["cost_per_token_input"],
        "cost_per_token_output": internal_config[0]["models"][3][
            "cost_per_token_output"
        ],
    },
    "gpt-3.5-turbo-0125": {
        "tokenizer": bpe_tokenize_encode,
        "cost_per_token_input": internal_config[0]["models"][4]["cost_per_token_input"],
        "cost_per_token_output": internal_config[0]["models"][4][
            "cost_per_token_output"
        ],
    },
    "Llama-2-13b-hf": {
        "tokenizer": bpe_tokenize_encode,
        "cost_per_token_input": internal_config[1]["models"][6]["cost_per_token_input"],
        "cost_per_token_output": internal_config[1]["models"][6][
            "cost_per_token_output"
        ],
    },
    "Llama-2-13b-chat-hf": {
        "tokenizer": bpe_tokenize_encode,
        "cost_per_token_input": internal_config[1]["models"][4]["cost_per_token_input"],
        "cost_per_token_output": internal_config[1]["models"][4][
            "cost_per_token_output"
        ],
    },
    "Llama-2-70b-hf": {
        "tokenizer": bpe_tokenize_encode,
        "cost_per_token_input": internal_config[1]["models"][10][
            "cost_per_token_input"
        ],
        "cost_per_token_output": internal_config[1]["models"][10][
            "cost_per_token_output"
        ],
    },
    "Llama-2-70b-chat-hf": {
        "tokenizer": bpe_tokenize_encode,
        "cost_per_token_input": internal_config[1]["models"][8]["cost_per_token_input"],
        "cost_per_token_output": internal_config[1]["models"][8][
            "cost_per_token_output"
        ],
    },
    "command-r": {
        "tokenizer": bpe_tokenize_encode,
        "cost_per_token_input": internal_config[3]["models"][0]["cost_per_token_input"],
        "cost_per_token_output": internal_config[3]["models"][0][
            "cost_per_token_output"
        ],
    },
    "command": {
        "tokenizer": bpe_tokenize_encode,
        "cost_per_token_input": internal_config[3]["models"][1]["cost_per_token_input"],
        "cost_per_token_output": internal_config[3]["models"][1][
            "cost_per_token_output"
        ],
    },
    "command-light": {
        "tokenizer": bpe_tokenize_encode,
        "cost_per_token_input": internal_config[3]["models"][2]["cost_per_token_input"],
        "cost_per_token_output": internal_config[3]["models"][2][
            "cost_per_token_output"
        ],
    },
    "command-nightly": {
        "tokenizer": bpe_tokenize_encode,
        "cost_per_token_input": internal_config[3]["models"][3]["cost_per_token_input"],
        "cost_per_token_output": internal_config[3]["models"][3][
            "cost_per_token_output"
        ],
    },
    "command-light-nightly": {
        "tokenizer": bpe_tokenize_encode,
        "cost_per_token_input": internal_config[3]["models"][4]["cost_per_token_input"],
        "cost_per_token_output": internal_config[3]["models"][4][
            "cost_per_token_output"
        ],
    },
    "mistral-7b-v0.1": {
        "tokenizer": bpe_tokenize_encode,
        "cost_per_token_input": internal_config[2]["models"][0]["cost_per_token_input"],
        "cost_per_token_output": internal_config[2]["models"][0][
            "cost_per_token_output"
        ],
    },
    "mistral-7b-instruct-v0.2": {
        "tokenizer": bpe_tokenize_encode,
        "cost_per_token_input": internal_config[2]["models"][1]["cost_per_token_input"],
        "cost_per_token_output": internal_config[2]["models"][1][
            "cost_per_token_output"
        ],
    },
    "mixtral-8x7b-instruct-v0.1": {
        "tokenizer": bpe_tokenize_encode,
        "cost_per_token_input": internal_config[2]["models"][2]["cost_per_token_input"],
        "cost_per_token_output": internal_config[2]["models"][2][
            "cost_per_token_output"
        ],
    },
    "text-bison": {
        "tokenizer": vertexai_encode,
        "cost_per_token_input": internal_config[4]["models"][0]["cost_per_token_input"],
        "cost_per_token_output": internal_config[4]["models"][0][
            "cost_per_token_output"
        ],
    },
    "chat-bison": {
        "tokenizer": vertexai_encode,
        "cost_per_token_input": internal_config[4]["models"][2]["cost_per_token_input"],
        "cost_per_token_output": internal_config[4]["models"][2][
            "cost_per_token_output"
        ],
    },
    "gemini-pro": {
        "tokenizer": vertexai_encode,
        "cost_per_token_input": internal_config[4]["models"][1]["cost_per_token_input"],
        "cost_per_token_output": internal_config[4]["models"][1][
            "cost_per_token_output"
        ],
    },
    "code-bison": {
        "tokenizer": vertexai_encode,
        "cost_per_token_input": internal_config[4]["models"][3]["cost_per_token_input"],
        "cost_per_token_output": internal_config[4]["models"][3][
            "cost_per_token_output"
        ],
    },
    "codechat-bison": {
        "tokenizer": vertexai_encode,
        "cost_per_token_input": internal_config[4]["models"][4]["cost_per_token_input"],
        "cost_per_token_output": internal_config[4]["models"][4][
            "cost_per_token_output"
        ],
    },
    "code-gecko": {
        "tokenizer": vertexai_encode,
        "cost_per_token_input": internal_config[4]["models"][5]["cost_per_token_input"],
        "cost_per_token_output": internal_config[4]["models"][5][
            "cost_per_token_output"
        ],
    },
    "claude-3-opus-20240229": {
        "tokenizer": ClaudeAdapter(api_key=ANTHROPIC_API_KEY, model="claude-3-opus-20240229").tokenize,
        "cost_per_token_input": internal_config[5]["models"][0]["cost_per_token_input"],
        "cost_per_token_output": internal_config[5]["models"][0]["cost_per_token_input"],
        },
    "claude-3-sonnet-20240229": {
        "tokenizer": ClaudeAdapter(api_key=ANTHROPIC_API_KEY, model="claude-3-sonnet-20240229").tokenize,
        "cost_per_token_input": internal_config[5]["models"][1]["cost_per_token_input"],
        "cost_per_token_output": internal_config[5]["models"][1]["cost_per_token_input"],
        },
    "claude-3-haiku-20240307": {
        "tokenizer": ClaudeAdapter(api_key=ANTHROPIC_API_KEY, model="claude-3-haiku-20240307").tokenize,
        "cost_per_token_input": internal_config[5]["models"][2]["cost_per_token_input"],
        "cost_per_token_output": internal_config[5]["models"][2]["cost_per_token_input"],
        },
}


def call_models(prompt: str, openai: OpenAIAdapter, llmproxy: LLMProxy) -> dict:
    start_openai = time.perf_counter()
    openai_output = openai.get_completion(prompt=prompt)
    end_openai = time.perf_counter()
    openai_latency = end_openai - start_openai

    openai_cost = calculate_estimated_max_cost(
        price_data={
            "prompt": model_info["gpt-4"]["cost_per_token_input"],
            "completion": model_info["gpt-4"]["cost_per_token_output"],
        },
        num_of_input_tokens=openai.tokenize(prompt=prompt)[0],
        max_output_tokens=openai.tokenize(prompt=openai_output)[0],
    )

    start_llmproxy = time.perf_counter()
    llmproxy_output = llmproxy.route(prompt=prompt)
    end_llmproxy = time.perf_counter()
    llmproxy_latency = end_llmproxy - start_llmproxy

    llmproxy_cost = calculate_estimated_max_cost(
        price_data={
            "prompt": model_info[llmproxy_output.response_model][
                "cost_per_token_input"
            ],
            "completion": model_info[llmproxy_output.response_model][
                "cost_per_token_output"
            ],
        },
        num_of_input_tokens=model_info[llmproxy_output.response_model]["tokenizer"](
            prompt=prompt
        )[0],
        max_output_tokens=model_info[llmproxy_output.response_model]["tokenizer"](
            prompt=llmproxy_output.response
        )[0],
    )

    return {
        "openai": {
            "response": openai_output,
            "latency": openai_latency,
            "cost": openai_cost,
        },
        "llmproxy": {
            "response": llmproxy_output.response,
            "latency": llmproxy_latency,
            "cost": llmproxy_cost,
        },
    }


sample_prompts = [
    """Congress enacts a $100 tax on the sale of any handgun to a private individual not for use in law enforcement or military duties. Will this new handgun tax survive a constitutional challenge?
a) Yes, if Congress could have banned possession of handguns outright.
b) Yes, if the dominant intent of Congress was that the tax would produce revenue.
c) No, if the tax does not result in a significant collection of revenue.
d) No, because the tax is clearly intended as a penalty on handgun ownership.""",
    """A salad dressing is made by combining 2 parts vinegar with 5 parts oil. How many ounces of oil should be mixed with 9 ounces of vinegar?
a) 2
b) 3.6
c) 22.5
d) 63""",
    """Which of the following states of matter is characterized by a closely packed arrangement of particles, resulting in a stable, definite shape and definite volume?
a) Solid
b) Liquid
c) Gas
d) Plasma""",
    # """The term Schwarzschild radius usually describes properties of ...
    # a) red dwarfs.
    # b) pulsars.
    # c) black holes.
    # d) galaxies.""",
    # """The Rosenhan study of mental institutions showed that
    # a) treatment at private institutions tends to be better than treatment at public institutions.
    # b) men are diagnosed at higher rates than women reporting the same symptoms.
    # c) it is difficult to convince medical professionals that one has a disorder when one does not.
    # d) confirmation bias may influence clinicians' views and treatments of mental patients.""",
    # """Children who are still sucking their thumbs when they enter kindergarten are often subject to teasing and ridicule for this behavior. What type of social sanction is applied in this instance?
    # a) Formal positive sanction
    # b) Formal negative sanction
    # c) Informal positive sanction
    # d) Informal negative sanction""",
    # """Who was the first US president to resign from that office?
    # a) Martin Van Buren
    # b) Andrew Jackson
    # c) Andrew Johnson
    # d) Richard Nixon""",
    # """A professor was the record owner in fee simple absolute of a 30-acre tract of land located in a small town. The professor made a conveyance thereof in these words, "I hereby convey my 30-acre tract to both my friend and my co-worker as joint tenants with right of survivorship. "Two years after making the conveyance to the friend and the co-worker, the friend executed a mortgage on the 30-acre tract of land to a bank to secure a loan. One year after executing the mortgage, the friend died intestate, leaving his uncle as his only heir. At the time of the friend's death, the indebtedness had not been paid to the bank. The jurisdiction in which the 30-acre tract of land is located recognizes a title theory of mortgages. In an appropriate action, the court should determine that title to the 30-acre tract of land is vested
    # a) in the co-worker, with the entire interest subject to the mortgage.
    # b) in the co-worker, free and clear of the mortgage.
    # c) half in the co-worker and half in the uncle, with both subject to the mortgage.
    # d) half in the co-worker, free of the mortgage, and half in the uncle, subject to the mortgage.""",
    # """A buffer is made from equal concentrations of a weak acid and its conjugate base. Doubling the volume of the buffer solution by adding water has what effect on its pH?
    # a) It has little effect.
    # b) It significantly increases the pH.
    # c) It significantly decreases the pH.
    # d) It changes the pH asymptotically to the pKa of the acid.""",
    # """For which of these two scenarios does the main character (who uses I/me/my) do something clearly morally wrong, according to ordinary moral standards in the US as of 2020? Scenario 1 | I brushed my daughter's teeth with drain cleaner. Scenario 2 | I had my old dog put down because I wanted a puppy.
    # a) Wrong, Wrong
    # b) Wrong, Not wrong
    # c) Not wrong, Wrong
    # d) Not wrong, Not wrong""",
    # """Some group (G, 0) is known to be abelian. Then which one of the following is TRUE for G?
    # a) g = g^-1 for every g in G
    # b) g = g^2 for every g in G
    # c) (g o h)^2 = g^2 o h^2 for every g,h in G
    # d) G is of finite order""",
    # """What is the percentage of children aged 13-15 in Japan who reported being bullied at least once in the past couple of months as of 2015?
    # a) 6%
    # b) 16%
    # c) 36%
    # d) 66%""",
    # """During his first therapy session, a client tells Dr. Goodwill that he wants to pay for therapy in cash and doesn't want Dr. Goodwill to keep a record of the therapy sessions or his payments. If Dr. Goodwill agrees to this arrangement she has acted:
    # a) ethically as long as she ascertains that the client's reasons for making this request are valid.
    # b) ethically since it is up to Dr. Goodwill to decide what to include in her therapy records.
    # c) ethically and legally as long as she includes the client's payments as income on her income tax forms.
    # d) unethically and possibly illegally.""",
    # """Statement 1 | For any two groups G and G', there exists a homomorphism of G into G'. Statement 2 | Every homomorphism is a one-to-one map.
    # a) True, True
    # b) False, False
    # c) True, False
    # d) False, True""",
    # """Low frequency supply is necessary for direct core type induction furnaces because
    # a) With normal frequeny supply the electromagnetic forces causes severe stirring action in the molten metal.
    # b) Magnetic coupling between the primary and secondary circuit is poor.
    # c) Both A and B.
    # d) None of the above.""",
    # """Which of the following problems can be solved by a standard greedy algorithm?
    # a) I. Finding a minimum spanning tree in an undirected graph with positive-integer edge weights
    # b) II. Finding a maximum clique in an undirected graph
    # c) III. Finding a maximum flow from a source node to a sink node in a directed graph with positive-integer edge capacities
    # d) I only
    # e) II only
    # f) III only
    # g) I and II only""",
    # """At what angle to the horizontal should an ideal projectile be launched so that its horizontal displacement (the range) is equal to its maximum vertical displacement?
    # a) sin–1 (1/g)
    # b) cos–1 (1/g)
    # c) 45°
    # d) tan–1 4""",
    # """For which of these two scenarios does the main character (who uses I/me/my) do something clearly morally wrong, according to ordinary moral standards in the US as of 2020? Scenario 1 | I used up the toilet paper in the bathroom that I share with my roommates after defecating, but I did not take a new roll from the closet to replace the old roll because there were none left. Scenario 2 | I took pictures of the women walking on the runway.
    # a) Wrong, Wrong
    # b) Wrong, Not wrong
    # c) Not wrong, Wrong
    # d) Not wrong, Not wrong""",
    # """Which of the following is not one of the categories of research design?
    # a) Exploratory research.
    # b) Descriptive research.
    # c) Causal research.
    # d) Desk research.""",
    # """In the diamond structure of elemental carbon, the nearest neighbors of each C atom lie at the corners of a
    # a) square
    # b) hexagon
    # c) cube
    # d) tetrahedron""",
    # """Determine the H+(aq) concentration in 1.0 M phenol, C6H5OH, solution. (The Ka for phenol is 1 × 10^-10.)
    # a) 1 × 10^-10 M
    # b) 1 × 10^-9 M
    # c) 1 × 10^-3 M
    # d) 1 × 10^-5 M""",
    # """Which of the following is the best explanation of the fallacy of loaded language?
    # a) Using emotionally charged language to create an impression about the subject of a claim, without making an argument that the language fits the subject.
    # b) Referring to an act committed by an opponent in negative terms while referring to the same act committed by the arguer or supporters in favorable terms.
    # c) Using language and punctuation in a way that a statement can have multiple interpretations, so it's not really clear what is meant.
    # d) Confusing figurative language with literal language""",
    # """Which of the following statements about plant sources of amino acids in human nutrition is correct?
    # a) All plant protein sources are deficient in essential amino acids
    # b) All plant protein sources contain all essential amino acids although some may be limited by the amount of particular amino acids
    # c) All plant protein sources are deficient in lysine
    # d) All plant protein sources are deficient in the sulphur amino acids acids""",
    # """A hard-edge painting is most likely to be characterized by
    # a) an even, solid paint application
    # b) blurry color mixed on the painting's surface
    # c) scratchy brush marks clearly separated
    # d) translucent multiple layers of paint""",
    # """Traveling at an initial speed of 1.5 × 10^6 m/s, a proton enters a region of constant magnetic field, B, of magnitude 1.0 T. If the proton's initial velocity vector makes an angle of 30° with the direction of B, compute the proton's speed 4 s after entering the magnetic field.
    # a) 5.0 × 10^5 m/s
    # b) 7.5 × 10^5 m/s
    # c) 1.5 × 10^6 m/s
    # d) 3.0 × 10^6 m/s""",
    # """This question refers to the following information.
    # a) "Lincoln was strongly anti-slavery, but he was not an abolitionist or a Radical Republican and never claimed to be one. He made a sharp distinction between his frequently reiterated personal wish that 'all men everywhere could be free' and his official duties as a legislator, congressman, and president in a legal and constitutional system that recognized the South's right to property in slaves. Even after issuing the Emancipation Proclamation he continued to declare his preference for gradual abolition. While his racial views changed during the Civil War, he never became a principled egalitarian in the manner of abolitionists such as Frederick Douglass or Wendell Phillips or Radical Republicans like Charles Sumner."
    # b) —Eric Foner, The Fiery Trial, 2010
    # c) How did President Lincoln's issuance of the Emancipation Proclamation alter the course of the Civil War?
    # d) The war came to a swift conclusion because the Proclamation made the Confederacy realize the futility of their cause.
    # e) The war grew in scope because the Proclamation caused Great Britain to join the fight on the side of the Union.
    # f) President Jefferson Davis of the Confederacy vowed massive resistance to any Union effort to free the slaves.
    # g) The war aims of the United States were no longer exclusively focused on the preservation of the Union.""",
    # """The state charged the accused with the intentional murder of a former girlfriend. He admitted to killing her, but asserted that he lacked the intent to murder, due to his suffering from a personality disorder complicated by two days of drinking alcohol heavily. The trial judge instructed the jury that "the law presumes that a person intends the ordinary consequences of his voluntary acts." The judge did not tell them that they had a choice or could infer another conclusion; they were told only that the law presumed it. The accused was convicted of depraved heart murder and sentenced to 100 years in prison. On appeal, the appellant argued that the court took the fact-finding task of determining criminal intent out of the jury's hands by in effect ordering them to find intent. The jury could have felt that they were being told that they had no other choice. The state supreme court affirmed the conviction, but the United States Supreme Court granted certiorari. Will the Court likely reverse the conviction?
    # a) Yes, because the wording of the instruction could have been viewed by jurors as a mandatory direction to find that there was intent to murder.
    # b) Yes, because the wording in effect caused the defendant to be a witness against himself by having him intend everything that he did.
    # c) No, because the defendant could raise an argument to rebut the permissible inference created by the jury instruction.
    # d) No, because the instruction was only a presumption that the jury did not have to consider in its deliberations.""",
    # """Solve 18.5 dol over m gal = 3.60 dol over 7.5 gal. Round to the nearest hundredth if necessary.
    # a) 8.86
    # b) 42.25
    # c) 32.54
    # d) 38.44""",
    # """If the Federal Reserve was concerned about the "crowding-out" effect they could engage in
    # a) expansionary monetary policy by lowering the discount rate.
    # b) expansionary monetary policy by selling Treasury securities.
    # c) contractionary monetary policy by raising the discount rate.
    # d) contractionary monetary policy by lowering the discount rate.""",
    # """Which country has undergone the most rapid nutrition transition in the last two decades?
    # a) China
    # b) Sweden
    # c) Britain
    # d) India""",
    # """A local politician starts a task force to reduce prejudice and hate crimes, and the task force provides recommendations for increased access to education, paid community improvement projects open to people from different, possibly conflicted, groups, and stricter anti-hate ordinances to be passed at the local level. These recommendations are based on:
    # a) I. Self-esteem hypothesis
    # b) II. Contact hypothesis
    # c) III. Hypothesis
    # d) IV. Legal hypothesis
    # e) I, II, and III
    # f) II, III, and IV
    # g) I, III, and IV
    # h) I, II, and IV""",
    # """Which of the following structures is most susceptible to damage by acid precipitation?
    # a) A monument made of granite
    # b) A roof made of slate
    # c) A tombstone made of marble
    # d) A statue made of gabbro""",
    # """For which of these two scenarios does the main character (who uses I/me/my) do something clearly morally wrong, according to ordinary moral standards in the US as of 2020? Scenario 1 | I spent the afternoon filling the pinata with razors for the birthday party. Scenario 2 | I used a syringe to treat cancer with radioactive polonium.
    # a) Wrong, Wrong
    # b) Wrong, Not wrong
    # c) Not wrong, Wrong
    # d) Not wrong, Not wrong""",
    # """Which of the following is another name for evading the issue?
    # a) "you too" fallacy
    # b) irrelevant conclusion
    # c) appeal to fear
    # d) argument against the person""",
    # """What is the biggest risk factor for infection with Ebola?
    # a) Working in a category IV laboratory
    # b) Attending a funeral of a victim outside
    # c) Nursing a patient at home
    # d) Attending football matches or large gatherings""",
    # """For which of these two scenarios does the main character (who uses I/me/my) do something clearly morally wrong, according to ordinary moral standards in the US as of 2020? Scenario 1 | It's hot in my house, so I give my visiting mother-in-law a thin cotton sheet. Scenario 2 | In the middle of the exam, I asked permission to go to the bathroom, then went to the bathroom, then returned to finish the exam.
    # a) Wrong, Wrong
    # b) Wrong, Not wrong
    # c) Not wrong, Wrong
    # d) Not wrong, Not wrong""",
    # """A laboratory manufactures nitroglycerin (a heavy, oily, explosive, poisonous liquid used chiefly in making dynamite) at its main plant. An artist who specialized in making delicate glass sculptures had a studio two blocks away from the laboratory's plant. Late one evening, there was an explosion at the laboratory's plant. The force of the explosion caused the artist's studio to be shaken, which resulted in the destruction of valuable artwork in the studio. The artist now asserts a tort action against the laboratory to recover damages. Which of the following, if established, would furnish the laboratory with its best possible defense?
    # a) The laboratory used extraordinary care in the manufacture and storage of nitroglycerin and was not guilty of any negligence that was causally connected with the explosion.
    # b) The laboratory has a contract with the federal government whereby all the nitroglycerin manufactured at its plant is used in U. S. military weapons systems.
    # c) The explosion was caused when lightning (an act of God) struck the plant during an electrical storm.
    # d) The harm that the artist suffered would not have resulted but for the abnormal fragility of the artist's work.""",
    # """The role of gathering and interpreting intelligence about foreign countries in order to allow policymakers to make good foreign policy decisions was given to
    # a) the Central Intelligence Agency (CIA).
    # b) the Federal Bureau of Investigation (FBI).
    # c) the National Security Council.
    # d) Both A and B are correct.""",
    # """A solid sphere (I = 0.06 kg·m^2) spins freely around an axis through its center at an angular speed of 20 rad/s. It is desired to bring the sphere to rest by applying a friction force of magnitude 2.0 N to the sphere’s outer surface, a distance of 0.30 m from the sphere’s center. How much time will it take the sphere to come to rest?
    # a) 4 s
    # b) 2 s
    # c) 0.06 s
    # d) 0.03 s""",
    # """If the demand for our exports rises while our tastes for foreign goods falls off then
    # a) the value of the dollar will tend to appreciate.
    # b) the value of the dollar will tend to depreciate.
    # c) exchange rates will be affected but not the value of the dollar.
    # d) the exchange rate will not be affected."""
]

openai = OpenAIAdapter(model="gpt-4", api_key=OPENAI_API_KEY)
llmproxy = LLMProxy(route_type="cost")

responses = dict()
for prompt in sample_prompts:
    responses[prompt] = call_models(
        prompt=prompt, openai=openai, llmproxy=llmproxy
    )

print(responses)
