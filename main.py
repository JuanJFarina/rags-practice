import faiss
import pandas as pd
from sentence_transformers import SentenceTransformer
from langchain_community.chat_models import ChatOllama
from langchain.agents import create_vectorstore_agent
from langchain.agents.agent_toolkits import VectorStoreToolkit, VectorStoreInfo
from langchain_core.vectorstores import VectorStore
from langchain_core.documents import Document
from typing import List, Tuple

class FaissVectorStoreWrapper(VectorStore):
    def __init__(self, faiss_index, df, embedding_model):
        self.faiss_index = faiss_index
        self.df = df
        self.embedding_model = embedding_model

    def add_texts(self, texts: List[str]) -> None:
        embeddings = self.embedding_model.encode(texts)
        self.faiss_index.add(embeddings)
        new_data = pd.DataFrame({'text': texts})
        self.df = pd.concat([self.df, new_data], ignore_index=True)

    @classmethod
    def from_texts(cls, texts: List[str], embedding_model: SentenceTransformer) -> 'FaissVectorStoreWrapper':
        embeddings = embedding_model.encode(texts)
        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(embeddings)
        df = pd.DataFrame({'text': texts})
        return cls(index, df, embedding_model)

    def similarity_search(self, query: str, top_k: int = 5) -> List[Tuple[str, float]]:
        query_vector = self.embedding_model.encode([query])
        distances, indices = self.faiss_index.search(query_vector, top_k)
        search_results = [(self.df.iloc[idx]['text'], distances[0][i]) for i, idx in enumerate(indices[0])]
        return [Document(page_content=result[0], metadata={"score": result[1]}) for result in search_results]


data = {
    "id": list(range(1, 38)),
    "text": [
        "Tendu means 'tight' or 'stretched', and simple literally means 'simple' or 'easy'.",
        "The battement tendu simple is the most important exercise in classical training: it provides significant development of all the objectives of the same; rotation, stability, balance, stretching of the foot and leg.",
        "It provides lifting power to demi-pointe, pointe, and jumps, as well as a correct base for turns, and an initial development of leg lifting strength.",
        "It is also an important movement in learning different positions, poses, and coordination of the head, torso, arms, legs, etc. Besides all the above, it is the first and main movement that studies the transition between a fundamental and a derived position.",
        "These battements are the foundation of all dance. They were found so brilliantly that it seems their creator penetrated into the very essence of the structure and functions of the ligamentous apparatus of the leg.",
        "A simple example from the daily life of a dancer demonstrates this. When a dancer slightly twists their leg while dancing and cannot step on it due to discomfort, as soon as they carefully perform battements tendus, the leg easily regains its working capacity.",
        "It is no wonder that it is also customary to do battements tendus before dancing to ‘warm up the legs’. But the legs not only warm up with this movement, they also get into a completely 'well-educated' state for the activity ahead, especially in allegro.",
        "When you see that the leg is not working well, it is easy to guess that the dancer was not strictly educated with battements tendus.",
        "It can be said with certainty that the battement tendu is the main element through which the correct extension of the entire leg from the knee to the tips of the toes is achieved. And this is a very important point because it is literally found in all movements of classical dance, especially in the progressive ones.",
        "The second exercise: battements tendus, combined with battements jetés, is where leg tension, rotation, and all small and large muscle groups are actively introduced into work. These basic movements train and develop leg strength.",
        "The purpose of this exercise is to achieve dexterity and ease of leg movement, which must be strongly extended with a curved arch of the foot. When performing this movement, the common extensor muscles, as well as the leg extensor muscles, are trained, thereby strengthening them.",
        "During this exercise, the legs get used to moving independently of the body, developing as if they were self-sufficient, their movements acquire lightness and ease. The peculiarity of this exercise is that the transition of the leg from an open to a closed position also occurs rhythmically, making the leg movements safe and extremely precise.",
        "The nature of the movement can be reduced to a preparation of the legs for performing dance steps, where the movements of one and the other leg are completely independent of each other.",
        "To achieve maximum benefit from this exercise, it is important to remember that the leg must move independently of the body, without the body wobbling, altering, or moving in any way (stability); the weight change, as we have mentioned earlier, should be the minimum necessary according to the strength and physical structure of each dancer so that they can move correctly, ideally being minimal or almost nonexistent.",
        "This battement can be performed towards the IV position forward, towards the II position, and towards the IV position backward; in all directions, it is important that the foot slides through a small perfectly rotated fundamental position before the heel leaves the floor to pass through demi-pointe and finally stretch as far as possible in pointe, with the arch of the foot strongly pronounced, the toes long, and again, a complete rotation.",
        "First, they are studied in 8/4 without pauses, later in 8/4 with pauses, and similarly in 4/4, 2/4, 1/4, and 1/8, being able to be performed in face and poses and also in tournant.",
        "In closing a battement tendu simple in I or V position, it is important to understand that the position fermée should not be struck, but rather the foot should return pressing the floor through the fundamental ouverte position and then the I or V position, joining both legs together, again without striking.",
        "As a general rule, bending either knee is unacceptable, as this, as previously discussed, allows rotational movements that could injure ligaments and tendons, in addition to not working correctly on rotation, stability, or posture; however, it should be noted that the main thing is to properly support the foot in the position fermée.",
        "In addition to the importance of how to adequately return to the position fermée in the battements tendus simples, it is also vital to open towards the position ouverte, especially because not only is the working leg important, but the supporting leg as well: it should take advantage of this instance to stretch and rotate more if possible, so that the subsequent position fermée is better, and improves even more with each battement tendu.",
        "The moment of opening a battement tendu is the moment to rotate the supporting leg more, the moment of closing a battement tendu is the moment to rotate the working leg more.",
        "Battements tendus simples are used in a large number of exercises; in warm-up, they are a vital part of it, performed especially at the side; in pliés, they may appear for more rotation work and muscle warm-up; in their own exercise, working on different variants; in ronds de jambe par terre, they can complement the rotation work; in grands battements jetés, they help maintain correct technique when opening and closing the large launches, and also allow for a higher tempo when interspersed to avoid excessive consecutive launches that could affect the exercise's performance.",
        "In the center, they are a primary exercise frequently performed in all methodologies or schools, where they also become a very important exercise for studying positions in face as well as poses and variants, and for coordinating port de bras with different movements of the arms, head, and torso, while performing various battements tendus.",
        "We will once again highlight the importance of the battement tendu simple by saying that the quality of it will determine the maximum quality of the rest of the movements to be learned and that it is, therefore, of utmost importance to develop this movement as perfectly as possible."
        "Patterns: These are the direction patterns we use in exercise sequences. It is another significant variable in the composition of exercises and in the development of the student; the study must be progressive and very careful to avoid unnecessarily complicating the exercises. The following are patterns specifically for barre exercises:",
        "Basic Pattern 1. The first movement pattern is simply using a single direction: in this case, the methodology indicates that one should always start by studying to the side, then forward, and finally backward.",
        "Basic Pattern 2. The second movement pattern is the famous en croix, a movement pattern that consists of performing a movement forward, side, backward, and side. This pattern can also be used in the form of en dehors and en dedans, starting backward for the latter.",
        "Basic Pattern 3. The third movement pattern involves 'shortening' the en croix so that it is only performed forward, side, and backward, performing some variant of the movement instead of going to the side. For example: 'In eight measures of 4/4 in V position, perform four battements tendus jetés forward, to the side, and backward in two counts each, and then perform a battement relevé lent in eight counts to the side.' This pattern can also be used in the form of en dehors and then en dedans, starting the sequence backward.",
        "Basic Pattern 4. The fourth movement pattern involves 'shortening' the en croix even more so that it is only performed forward and to the side, performing a variant of the movement before continuing with the second half of the en croix (thus ending again with the variant). For example: 'In four measures of 4/4 in V position, perform two battements fondus forward and two to the side in two counts each, then close V position and perform a III port de bras, then continue with the en dedans part of the exercise also ending with III port de bras.'",
        "Basic Pattern 5. The fifth movement pattern involves performing a movement forward, then performing the same movement backward, and then to the side. Similarly, repeat backward, then forward, and then to the side again. For example: 'In four measures of 4/4 in V position, perform two battements tendus simples forward in one count each and in two counts perform a third one making passé par terre backward and closing V position. Repeat backward with passé par terre forward and closing V position, then perform four battements tendus simples to the side in one count each and two battements tendus pour le pied in two counts each.'",
        "Pattern with Poses 6. The sixth movement pattern is the beginning of the study of poses, performed according to pattern 2 en croix in effacée forward, side, effacée backward, and side. This pattern is also combinable with the third and fourth patterns.",
        "Pattern with Poses 7. The seventh movement pattern continues the study of poses also according to pattern 2 en croix, performed forward, to the écartée backward, backward, and to the écartée forward. This pattern is also combinable with the third and fourth patterns.",
        "Pattern with Poses 8. The eighth movement pattern concludes the study of poses also according to pattern 2 en croix, performed in effacée forward, écartée backward, effacée backward, and écartée forward. This pattern is also combinable with the third and fourth patterns.",
        "Pattern with Alternation 9. The ninth movement pattern is the beginning of using the inside leg: we take pattern 1 and perform with the outside leg only a movement to the side, ending in V position backward, after which we use the leg from the barre forward. Repeat to the side with the outside leg and then backward with the leg from the barre. For example: 'In four measures of 4/4, perform four battements tendus simples to the side in V position ending backward, then with the inside leg perform four battements tendus simples forward, in 1/4 each. Repeat to the side and backward with the inside leg.'",
        "Pattern with Alternation 10. The tenth movement pattern takes pattern 3, performing with the outside leg a movement forward, side, and backward, then with the inside leg a movement forward. Then repeat in en dedans, starting backward, side, and forward, and with the leg from the barre backward. For example: 'In thirty-two measures of 3/4, perform a battement soutenu at 90° forward, side, and backward, and then a battement relevé lent forward with the inside leg, in four measures each movement. Repeat in en dedans starting backward.'",
        "Pattern with Alternation 11. The eleventh movement pattern takes pattern 5, performed forward with the outside leg, backward with the inside leg, and to the side with the outside leg, finally starting again backward. For example: 'In eight measures of 2/4, perform two battements tendus jetés forward with the outside leg, two backward with the inside leg, in 1/4 each, and seven to the side with the outside leg, in 1/8 each. Then start in en dedans backward.'",
        "Pattern with Alternation 12. The twelfth movement pattern continues the study of the inside leg: forward with the outside leg, backward with the inside leg, to the side with the outside leg, and forward with the inside leg, then repeat starting backward. For example: 'In eight measures of 2/4 in V position, perform two grands battements jetés forward, backward, to the side, and forward with the inside leg in 1/4 each, then backward, forward, to the side, and backward, also in 1/4 each.'",
        "Complex Pattern 13. The thirteenth movement pattern will combine patterns with leg alternation and poses 6 and 9; first, perform a movement with the outside leg to the side, then with the inside leg in pose croisée forward, with the outside leg to the side, and with the inside leg in pose croisée backward. For example: 'In four measures of 4/4 in V position, perform seven battements tendus jetés to the side to I position in 1/8 each, closing the last one V position backward to then perform two jetés croisée forward in 1/4 each and then two piqués in 1/8 each, then repeat with croisée backward.'",
        "Complex Pattern 14. The fourteenth movement pattern will combine patterns 7 and 11; first, perform a movement with the outside leg forward, then with the inside leg backward, and finally with the outside leg in pose écartée backward. Then repeat starting backward. For example: 'In thirty-two measures of 3/4 in V position, perform two battements développés ballottés in four measures each (forward-backward, forward-backward), then perform a développé to the écartée backward and a détourné away from the barre in two measures, repeat développé and détourné, and again développé to the écartée backward, two ronds de jambe at 90° in one measure each, and close V position backward to start everything from backward.'"
    ]
}


df = pd.DataFrame(data)
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = embedding_model.encode(df['text'].tolist())

index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)
faiss.write_index(index, "vector_index.faiss")
df.to_csv("text_data.csv", index=False)
index = faiss.read_index("vector_index.faiss")
df = pd.read_csv("text_data.csv")

store = FaissVectorStoreWrapper(faiss_index=index, df=df, embedding_model=embedding_model)

store_info = VectorStoreInfo(
    name="FaissVectorStore on Battements Tendus Simples and Patterns for ballet exercises",
    description="A vector store using Faiss for indexing, containing information about the ballet exercise battements tendus simples and also patterns for creating ballet exercises.",
    vectorstore=store,
)

toolkit = VectorStoreToolkit(
    llm=ChatOllama(model="llama3"),
    vectorstore_info=store_info,
)

agent = create_vectorstore_agent(
    llm=ChatOllama(model="llama3"),
    toolkit=toolkit,
    verbose=True,
    handle_parsing_errors=True,
)

message = "Please create an easy battement tendu simple exercise at the barre for beginners"
result = agent.invoke(message)

print(f"{message = }")
print(f"answer = '{result['output']}'")
