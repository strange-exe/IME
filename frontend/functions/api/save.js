export async function onRequestPost(context) {
    try {
        // Parse the incoming JSON payload from the frontend
        const body = await context.request.json();
        const { category, records } = body;

        if (!category || !records || !Array.isArray(records) || records.length === 0) {
            return new Response(JSON.stringify({ success: false, error: "Invalid payload or empty records" }), {
                status: 400,
                headers: { "Content-Type": "application/json" }
            });
        }

        // Access the bound D1 database (assuming the binding is named 'DB')
        const db = context.env.DB;

        // Determine which table to insert into based on the category
        let statements = [];
        for (const record of records) {
            if (category === "tonnage") {
                statements.push(
                    db.prepare(
                        "INSERT INTO tonnage_records (vessel_name, account_name, open_port, open_date, vessel_type, vessel_size) VALUES (?, ?, ?, ?, ?, ?)"
                    ).bind(
                        record.vessel_name || "",
                        record.account_name || "",
                        record.open_port || "",
                        record.open_date || "",
                        record.vessel_type || "",
                        record.vessel_size || ""
                    )
                );
            } else if (category === "cargo_vc") {
                statements.push(
                    db.prepare(
                        "INSERT INTO vc_records (account_name, cargo_name, loading_port, discharge_port, laycan, cargo_type) VALUES (?, ?, ?, ?, ?, ?)"
                    ).bind(
                        record.account_name || "",
                        record.cargo_name || "",
                        record.loading_port || "",
                        record.discharge_port || "",
                        record.laycan || "",
                        record.cargo_type || ""
                    )
                );
            } else if (category === "cargo_tc") {
                statements.push(
                    db.prepare(
                        "INSERT INTO tc_records (account_name, cargo_name, delivery_port, redelivery_port, duration, laycan, cargo_type) VALUES (?, ?, ?, ?, ?, ?, ?)"
                    ).bind(
                        record.account_name || "",
                        record.cargo_name || "",
                        record.delivery_port || "",
                        record.redelivery_port || "",
                        record.duration || "",
                        record.laycan || "",
                        record.cargo_type || ""
                    )
                );
            }
        }

        // Execute all inserts in a batch
        if (statements.length > 0) {
            await db.batch(statements);
        }

        return new Response(JSON.stringify({ success: true, saved_count: statements.length }), {
            status: 200,
            headers: { "Content-Type": "application/json" }
        });

    } catch (err) {
        return new Response(JSON.stringify({ success: false, error: err.message }), {
            status: 500,
            headers: { "Content-Type": "application/json" }
        });
    }
}
