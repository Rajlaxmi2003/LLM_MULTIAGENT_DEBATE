import random

import openai
from flask import Flask, jsonify, render_template, request

openai.api_key = "ollama"
openai.api_base = "http://localhost:11434/v1"

app = Flask(__name__)


def parse_bullets(sentence):
    bullets_preprocess = sentence.split("\n")
    bullets = []

    for bullet in bullets_preprocess:
        try:
            idx = bullet.find(next(filter(str.isalpha, bullet)))
        except Exception:
            continue

        bullet = bullet[idx:]
        if bullet:
            bullets.append(bullet)

    return bullets


def generate_answer(answer_context):
    try:
        completion = openai.ChatCompletion.create(
            model="llama3.2",
            messages=answer_context,
            n=1,
        )
        return completion
    except Exception as exc:  # pragma: no cover - surfaced to the UI
        print("Actual error:", repr(exc))
        raise


def construct_message(agents, question, idx):
    if len(agents) == 0:
        return {
            "role": "user",
            "content": "Original question: {}\n\nCan you verify that your answer is correct? Please reiterate your answer, making sure to state your answer at the end of the response.".format(question),
        }

    prefix_string = "Original question: {}\n\nThese are the recent/updated opinions from other agents: ".format(question)

    for agent in agents:
        agent_response = agent[idx]["content"]
        response = "\n\n One agent response: ```{}```".format(agent_response)
        prefix_string = prefix_string + response

    prefix_string = prefix_string + "\n\nUse these opinions carefully as additional advice. Provide an updated answer and state it clearly at the end."
    return {"role": "user", "content": prefix_string}


def construct_assistant_message(completion):
    content = completion["choices"][0]["message"]["content"]
    return {"role": "assistant", "content": content}


def parse_answer(sentence):
    if not sentence:
        return None
    parts = sentence.split(" ")

    for part in parts[::-1]:
        try:
            return float(part)
        except ValueError:
            continue
    return None


def most_frequent(values):
    if not values:
        return None
    counter = 0
    num = values[0]

    for value in values:
        current_frequency = values.count(value)
        if current_frequency > counter:
            counter = current_frequency
            num = value

    return num


def build_equation(values):
    a, b, c, d, e, f = values
    return "{} + {} * {} + {} - {} * {}".format(a, b, c, d, e, f)


def evaluate_agent_debate(values=None, agents=2, rounds=3, question=None):
    if question:
        task_prompt = "Answer the following question as accurately as you can. Explain your reasoning and state your answer clearly at the end.\n\n{}".format(question)
        question_label = question
        correct_answer = None
    else:
        a, b, c, d, e, f = values
        task_prompt = "What is the result of {}+{}*{}+{}-{}*{}? Make sure to state your answer at the end of the response.".format(a, b, c, d, e, f)
        question_label = build_equation(values)
        correct_answer = a + b * c + d - e * f

    agent_contexts = [
        [{"role": "user", "content": task_prompt}]
        for _ in range(agents)
    ]

    for round_index in range(rounds):
        for i, agent_context in enumerate(agent_contexts):
            if round_index != 0:
                agent_contexts_other = agent_contexts[:i] + agent_contexts[i + 1 :]
                message = construct_message(agent_contexts_other, task_prompt, 2 * round_index - 1)
                agent_context.append(message)

            completion = generate_answer(agent_context)
            assistant_message = construct_assistant_message(completion)
            agent_context.append(assistant_message)

    text_answers = []
    responses = []

    for index, agent_context in enumerate(agent_contexts):
        raw_text = agent_context[-1]["content"]
        responses.append({
            "agent": index + 1,
            "response": raw_text,
            "parsed_answer": parse_answer(raw_text.replace(",", ".")),
        })
        parsed_answer = parse_answer(raw_text.replace(",", "."))
        if parsed_answer is not None:
            text_answers.append(parsed_answer)

    if correct_answer is None:
        final_answer = responses[-1]["response"]
        score = None
    else:
        final_answer = most_frequent(text_answers) if text_answers else None
        score = 1 if final_answer == correct_answer else 0

    return {
        "equation": question_label,
        "is_custom": question is not None,
        "correct_answer": correct_answer,
        "agents": agents,
        "rounds": rounds,
        "final_answer": final_answer,
        "score": score,
        "responses": responses,
    }


@app.get("/")
def index():
    return render_template("index.html")


@app.post("/api/run")
def run_debate_api():
    payload = request.get_json(silent=True) or {}

    try:
        agents = int(payload.get("agents", 2))
        rounds = int(payload.get("rounds", 3))
    except (TypeError, ValueError):
        return jsonify({"error": "Agents and rounds must be integers."}), 400

    if agents < 1 or rounds < 1:
        return jsonify({"error": "Agents and rounds must be at least 1."}), 400

    question = payload.get("question", "")
    if not isinstance(question, str):
        return jsonify({"error": "Your question must be text."}), 400
    question = question.strip()
    if len(question) > 4000:
        return jsonify({"error": "Your question must be 4,000 characters or fewer."}), 400

    values = []
    if question:
        values = None
    else:
        for key in ["a", "b", "c", "d", "e", "f"]:
            value = payload.get(key)
            if value is None or value == "":
                return jsonify({"error": f"Please provide {key.upper()} or enter a custom question."}), 400
            try:
                values.append(int(value))
            except (TypeError, ValueError):
                return jsonify({"error": f"{key.upper()} must be an integer."}), 400

    try:
        result = evaluate_agent_debate(values, agents=agents, rounds=rounds, question=question or None)
        return jsonify(result)
    except Exception as exc:  # pragma: no cover - surfaced to the UI
        return jsonify({"error": "Could not complete the debate: {}".format(exc)}), 500


@app.get("/api/random")
def random_equation():
    values = [random.randint(0, 29) for _ in range(6)]
    equation = build_equation(values)
    return jsonify({"values": values, "equation": equation})


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
