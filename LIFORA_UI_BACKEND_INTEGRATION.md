# Lifora Version 3 — UI + Backend Integration Patch

This is a **patch archive**, not a full project. Extract its contents into the root of your existing `lifora_version_3` folder and allow the listed files to be added/replaced.

## Files in this patch

| Action | Relative path |
|---|---|
| Replace | `frontend/index.html` |
| Replace | `frontend/src/main.jsx` |
| Replace | `frontend/src/App.jsx` |
| Add | `frontend/src/lifora-ui-theme.css` |
| Add | `frontend/public/lifora-mark.svg` |

No backend source file is replaced. Version 3 already contains the FastAPI endpoints used by this React frontend; preserving the existing backend avoids changing the emergency state machine, MongoDB schema, RAG service, authentication, or model integration.

## What is integrated

The supplied LIFORA_UI design language (blue/navy palette, logo, responsive cards, forms, tables and navigation styling) is adapted to the existing React app rather than importing its browser-only JavaScript.

The React app continues to use the real Version 3 API:

- Authentication: `POST /api/v1/auth/login`, `GET /api/v1/auth/me`
- Patient list/profile: `GET /api/v1/patients`, `GET /api/v1/patients/{patient_id}`
- Dashboard and monitoring: `/dashboard`, `/vitals`
- Symptoms, reports, history and medicines: existing patient-scoped endpoints
- Alerts and emergency workflow: `/alerts`, `/emergency`, `/emergency-events`, `/emergency-contacts`
- AI assistant: `POST /api/v1/patients/{patient_id}/explain`
- System health: `GET /api/v1/system/status`

Additional React screens expose live-refreshing recent vitals, a read-only patient record summary, an alert review list, and a dedicated Ask Lifora AI page.

## Deliberately not imported from the UI prototype

The original `LIFORA_UI` ZIP is explicitly a frontend-only demo. Its localStorage demo passwords, role switcher, simulated readings, and simulated call/SMS workflow are **not** used in the integrated app.

The current Version 3 backend does not expose user-management, care-team/consent, message-center, device-inventory, or separate doctor/technician portal APIs. Those prototype screens are therefore not represented as working backend features. The UI displays the actual server role and patient access returned by the backend; it does not invent accounts or bypass server authorization.

The existing backend still controls emergency rules and notifications. Twilio remains governed by the backend `.env` configuration and is disabled by default unless deliberately configured.

## Apply the patch

1. Make a backup of your `lifora_version_3` folder.
2. Extract this ZIP **into the root** of `lifora_version_3`, preserving folders. Allow the three `Replace` files to be overwritten.
3. Keep your existing root `.env`, MongoDB data, model artifacts, RAG indexes, and backend files unchanged.
4. In a terminal:

   ```bash
   cd lifora_version_3/frontend
   npm install
   npm run build
   ```

5. Start the backend and frontend using the commands in your existing `README_SETUP.md`.

The `main.jsx` import is corrected to use the existing `frontend/src/styles.css` filename. The new theme stylesheet is loaded after it, so the original component styling remains the base and the supplied UI theme overrides it.

## Configuration

Keep `frontend/.env` pointed at your backend, for example:

```env
VITE_API_BASE_URL=http://127.0.0.1:8000/api/v1
```

Do not put JWT secrets, database credentials, Groq keys, Twilio credentials, or device tokens in frontend environment variables.

## Validation performed on this patch

- React JSX/JavaScript syntax parsed successfully.
- Both CSS files passed CSS parser checks.
- The API paths used by the added screens exist in the supplied Version 3 FastAPI backend.
- A full Vite production build could not be completed in the Linux review environment because the uploaded `node_modules` contains Windows-only optional Rollup binaries. Run `npm install` and `npm run build` on your development machine after applying the patch.

## Security note about the supplied archive

The uploaded Version 3 archive contains a populated root `.env` file. This patch does not copy or overwrite it. If any values in that file are real credentials or keys, rotate them and keep `.env` private and out of shared ZIPs/source control.

This remains a prototype and is not a certified medical device or emergency-response service.
