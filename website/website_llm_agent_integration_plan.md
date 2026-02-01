Look at the initial code in the website directory. This is my current frontend code. I would like to fully integrate the zoning agent from LLM/exploration/zoning_agent.py To do this a couple of changes are needed. 

Firstly, right now it just gives a initial drop down where you can select your current cluster. This is the right idea by I want it so that whenever their is a clustering task (which will happen initially now), their will be a drop down list within the chat area, and you can pick one of the clusters. This should happen whenever a clustering task is initiated. The static drop down list should be removed.

Then, integrate the LLM agent into the chat window so that for the most part, you only give feedback and choose solutions through the chat window. Also be sure to integrate a loading widget over the map while the model generates a response. You can take some inspiration from LLM/exploration/run_agent.py. 

 I am running this on a linux research server, and for now everything can be in local host. Write your code into the website folder. Also remember to use uv for all python related tasks.

Create a plan for this project and ask me for approval of it. Focus on writing lower verbosity and clear code.