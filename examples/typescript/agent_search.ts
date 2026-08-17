/** Retrieve cited Athena context for a TypeScript AI agent. */

type Citation = {
  source_uri?: string;
  locator?: string;
  section?: string;
};

type SearchItem = {
  text: string;
  title?: string;
  citations?: Citation[];
};

type SearchResponse = { items: SearchItem[] };

export async function searchAthena(query: string, collectionId: string): Promise<SearchItem[]> {
  const baseUrl = (process.env.ATHENA_URL ?? "http://127.0.0.1:8080").replace(/\/$/, "");
  const token = process.env.ATHENA_API_TOKEN;
  if (!token) throw new Error("Set ATHENA_API_TOKEN before searching Athena.");

  const response = await fetch(`${baseUrl}/v1/search`, {
    method: "POST",
    headers: { Authorization: `Bearer ${token}`, "Content-Type": "application/json" },
    body: JSON.stringify({ query, collection_ids: [collectionId] }),
  });
  if (!response.ok) throw new Error(`Athena returned HTTP ${response.status}.`);
  return (await response.json() as SearchResponse).items;
}

export async function agentContext(query: string, collectionId: string): Promise<string> {
  const passages = await searchAthena(query, collectionId);
  if (!passages.length) return "No relevant Athena sources were found. Do not make up an answer.";

  return ["Use only the cited context below. If it is insufficient, say so.", ...passages.map((passage) => {
    const citation = passage.citations?.[0];
    const source = citation?.source_uri ?? passage.title ?? "unknown source";
    const location = citation?.locator ?? citation?.section ?? "source";
    return `\n[${source} — ${location}]\n${passage.text}`;
  })].join("\n");
}

if (import.meta.main) {
  const query = process.argv[2] ?? "What information is available?";
  const collection = process.argv[3] ?? "support";
  console.log(await agentContext(query, collection));
}
