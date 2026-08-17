# Agent Integration Examples

These examples use Athena's HTTP API directly, so they work with any agent
framework. They retrieve passages and citations that an agent can place in its
working context or final answer.

Set the Athena endpoint and token before running either example:

```bash
export ATHENA_URL=http://127.0.0.1:8080
export ATHENA_API_TOKEN=<your token>
```

The examples search the `support` collection by default. Pass another collection
as the second argument.

```bash
python examples/python/agent_search.py "How do returns work?" support
node --experimental-strip-types examples/typescript/agent_search.ts "How do returns work?" support
```

The output is formatted context, not a generated answer. Supply it to your agent
along with your own instructions, and retain the returned citations when the
agent responds.
