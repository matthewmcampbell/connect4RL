import json
from model.ddqn import DDQN

dummy = {
        "layers": (42,2,7),
        "memory_size": 1,
        "memory_cutoff": int(1 / 5),
        "target_update": 1,
        "epsilon": {"start": 0.95, "end": 0.05, "decay": 1000, "agent_train": 0, "burn_in": 0},
        "batch_size": 1,
        "gamma": 1,  # Between 0.65 and 0.75
        "name": "dummy",
        "gen": 0
    }

model = DDQN(**dummy)
model.load_model("build0003_gen11")

def predict(event, context):
    try:
        f_context = json.loads(event['body'])['obs']
        body = {
            "message": "Go Serverless v1.0! Your function executed successfully!",
            "input": event,
            "prediction": model.select_action(full_context=f_context, train=False)
        }

        response = {
            "statusCode": 200,
            "body": json.dumps(body),
            "headers": {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*',
                "Access-Control-Allow-Credentials": True
            },
        }
        return response

    except Exception as e:
        return {
            "statusCode": 500,
            "headers": {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*',
                "Access-Control-Allow-Credentials": True
            },
            "body": json.dumps({"error": repr(e), "full": json.loads(event['body'])})
        }