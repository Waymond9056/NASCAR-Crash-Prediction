## Inspiration
NASCAR: Maybe it's the randomness, maybe it's the fans, or maybe it's the inevitable crashes that take place in every race. Regardless, any avid fan of the sport will guarantee you that these moments are the most entertaining (or stressful if it is your favorite driver in the crash), and missing them would undoubtedly be regretful. On the other hand, NASCAR teams tend to anticipate these moments, looking for the ideal opportunity to protect their drivers or plan a pit stop to save time throughout the race.

## What it does
"The Caution Flag" is a model that utilizes data throughout NASCAR races to accurately predict when moments of high caution will occur, specifically crashes between drivers. From there, it can display live adaptive data of a "risk factor" (probability of a crash taking place during a lap) and the top three drivers that are most highly on the "watchlist" (have the highest probability of being a part of the accident. 

## How we built it
We created "The Caution Flag" through a neural network model that utilized large datasets of lap time data and crash time data to successfully train itself and lead to a user-friendly front-end output. Our data was manually filtered due to the limited nature of NASCAR data. Additionally, we used explainable AI to correlate the risk of a crash to the drivers throughout the race that are most likely to have been a part of the accident. Finally, through the use of the React framework, all of this information was displayed in a user interface that allows any NASCAR enjoyer to have a more pleasurable watching experience.

## Challenges we ran into
One of our biggest challenges throughout this project was the lack of data available related to NASCAR. While professional teams and partnered developers of NASCAR have access to ERDP data (Event Racing Data Platform), which happens to be significantly more detailed than any data provided to the public, openly available data related to NASCAR is very limited on the internet. Initially, we had to filter through data related to lap times in order to pinpoint the exact positions of certain vehicles done across numerous races.

## Accomplishments that we're proud of
While many other data analysis projects utilize clean, publically available data throughout their models, we are most proud of the resilience we exhibited as we sorted through numerous datasets with limited information. Prior research showed us that NASCAR tends to be a secretive company when it comes to providing its fans with adequate datasets, so being one of the first projects to utilize precise NASCAR race data gave us a feeling of uniqueness. 

## What we learned
Ultimately, this project was a gateway for our experiences in the collision of data analysis with machine learning models in Computer Science. Working with complex frameworks such as PyTorch or libraries such as Captum gave each of our team members a glimpse into our future of becoming software developers with endless career possibilities. 

## What's next for The Caution Flag
We hope to share "The Caution Flag" with the NASCAR world so that fans and professional teams alike can benefit from this model. Teams can utilize our open-source code and input their own private data into their models to assist their drivers during races, and the organization itself could utilize this data to provide a more pleasurable viewing experience for fans like us!
