import os
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.exceptions import OutputParserException
from dotenv import load_dotenv

load_dotenv()

class Chain:
    def __init__(self):
        self.llm = ChatGroq(temperature=0, groq_api_key=os.getenv("GROQ_API_KEY"), model_name="llama-3.1-70b-versatile")

    def extract_jobs(self, cleaned_text):
        prompt_extract = PromptTemplate.from_template(
            """
            ### SCRAPED TEXT FROM WEBSITE:
            {page_data}
            ### INSTRUCTION:
            The scraped text is from the career's page of a website.
            Your job is to extract the job postings and return them in JSON format containing the following keys: `role`, `experience`, `skills` and `description`.
            Only return the valid JSON.
            ### VALID JSON (NO PREAMBLE):
            """
        )
        chain_extract = prompt_extract | self.llm
        res = chain_extract.invoke(input={"page_data": cleaned_text})
        try:
            json_parser = JsonOutputParser()
            res = json_parser.parse(res.content)
        except OutputParserException:
            raise OutputParserException("Context too big. Unable to parse jobs.")
        return res if isinstance(res, list) else [res]

    def write_mail(self, job):
        prompt_email = PromptTemplate.from_template(
            """
            ### JOB DESCRIPTION:
            {job_description}

            ### INSTRUCTION:
           Write a professional job application for a above job description using the following details:

Name: Nadeem Ahmad

Email: Ahmadnadeem6236@gmail.com

Phone: +91-9205526460

Location: New Delhi, India

Education:

Bachelor of Technology in Computer Science and Engineering (Artificial Intelligence), Jamia Hamdard, expected to graduate in July 2025.
Relevant coursework includes Object-Oriented Programming, Operating Systems, Database Management Systems, and System Design.
Skills:

Programming Languages: JavaScript, TypeScript, Python, Golang, C/C++.
Frameworks/Libraries: Node.js, Express.js, React.js, Next.js, FastAPI, Flask, Jest, Pytest.
Databases: MySQL, PostgreSQL, ElasticSearch, MongoDB.
Cloud: AWS (Lambda, API Gateway, S3, DynamoDB).
Tools: VSCode, Vim, Git/GitHub, MySQL Workbench, Atlas.
Experience:

Full Stack Intern at Candidate.live (July 2024 - October 2024):
Designed and implemented user interfaces, achieving an 80 per cent faster load time.
Supported back-end tasks, reducing latency by 20%.
Identified and resolved bugs, improving codebase organization by 60%.
Participated in code reviews.
Tech stack: ReactJS, NextJS, TailwindCSS, NodeJS, Express, PostgreSQL, and MongoDB.
Web Development Intern at AVIRO ENERGY (January 2024 - June 2024):
Worked on front-end development and created reusable components.
Delivered high-quality code adhering to performance and best practices.
Tech stack: ReactJS, TailwindCSS, Material-UI, NodeJS.
Achievements:

Completed "Data Structure and Algorithms in Python" by Coding Ninjas.
Finished Full Stack Open-Source Cohort 1 by 100xDEVS.
Create a personalized, enthusiastic cover letter highlighting the candidate's passion for software development, emphasizing their relevant skills, and expressing interest in contributing to the hiring company's projects.
            ### EMAIL (NO PREAMBLE):

            """
        )
        chain_email = prompt_email | self.llm
        res = chain_email.invoke({"job_description": str(job)})
        return res.content

if __name__ == "__main__":
    print(os.getenv("GROQ_API_KEY"))