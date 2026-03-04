// helper script to talk directly to the Chroma HTTP API from Node
// run with `node chroma_setup.js`


const BASE = 'http://localhost:8000/api/v2';

async function createTenant(name) {
  const res = await fetch(`${BASE}/tenants`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ name })
  });
  const text = await res.text();
  console.log('createTenant', res.status, text);
}

async function createDatabase(tenant, db) {
  const res = await fetch(`${BASE}/tenants/${tenant}/databases`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ name: db })
  });
  const text = await res.text();
  console.log('createDatabase', res.status, text);
}

async function createCollection(tenant, db, coll) {
  const res = await fetch(`${BASE}/tenants/${tenant}/databases/${db}/collections`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ name: coll, get_or_create: true })
  });
  const text = await res.text();
  console.log('createCollection', res.status, text);
}

(async () => {
  try {
    await createTenant('default');
    await createDatabase('default', 'default');
    await createCollection('default', 'default', 'products');
  } catch (err) {
    console.error(err);
  }
})();