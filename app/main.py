import json

import requests
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware

from app.setup import LINEAR_API_TOKEN, MODEL_PATH, ScopeGenerator

# Program
scope_gen = ScopeGenerator()
try:
    scope_gen.load(MODEL_PATH)
except Exception:
    print("Model loading failed.")
    pass


# API
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Routes
@app.post("/webhooks/linear/issues")
async def webhook(request: Request) -> dict[str, str]:
    """Add scope to newly created issue using its title."""
    # Parse request
    data = await request.json()
    if data["action"] != "create" or data["type"] != "Issue":
        return {"status": "ignored"}
    title = data["data"]["title"]

    # Run the program
    pred = scope_gen(title)
    description, acceptance_criteria, sub_tasks, assumptions, dependencies = (
        pred.description,
        pred.acceptance_criteria,
        pred.sub_tasks,
        pred.assumptions,
        pred.dependencies,
    )
    text = f"{description}\n{acceptance_criteria}\n{sub_tasks}\n{assumptions}\n{dependencies}"

    # Update issue
    query = f"""
    mutation IssueUpdate {{
        issueUpdate(
            id: "{data["data"]["id"]}"
            input: {{
                description: {json.dumps(text)}
            }}
        ) {{
            success
            issue {{
                id
                description
            }}
        }}
    }}
    """

    r = requests.post(
        "https://api.linear.app/graphql",
        json={"query": query},
        headers={"Content-Type": "application/json", "Authorization": LINEAR_API_TOKEN},
    )
    response = json.loads(r.content)

    if "errors" in response:
        return HTTPException(status_code=400, detail=response["errors"][0]["message"])
    return {"status": "success"}


def main():
    """Run API."""
    uvicorn.run(
        "app.main:app",
        reload=True,
        ssl_keyfile="./certificates/localhost+2-key.pem",
        ssl_certfile="./certificates/localhost+2.pem",
    )


if __name__ == "__main__":
    main()
