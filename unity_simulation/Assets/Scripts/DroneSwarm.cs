using UnityEngine;

[RequireComponent(typeof(Rigidbody))]
public class DroneSwarm : MonoBehaviour
{
    private Rigidbody rb;

    public Vector3 targetPosition;

    public float maxSpeed = 12f;
    public float steeringForce = 6f;
    public float hoverForce = 15f;
    public float damping = 3f;

    public float separationRadius = 2.5f;
    public float separationForce = 8f;

    private bool isFailed = false;

    void Start()
    {
        rb = GetComponent<Rigidbody>();
        rb.useGravity = true;
        rb.linearDamping = 1f;
    }

    void FixedUpdate()
    {
        if (isFailed) return;

        MoveDrone();
    }

    void MoveDrone()
    {
        Vector3 force = Vector3.zero;

        // --- Move Toward Target ---
        Vector3 desired = targetPosition - transform.position;
        Vector3 steer = desired - rb.linearVelocity;
        force += steer * steeringForce;

        // --- Separation ---
        Collider[] neighbors = Physics.OverlapSphere(transform.position, separationRadius);
        foreach (Collider col in neighbors)
        {
            if (col.gameObject != gameObject && col.GetComponent<DroneSwarm>())
            {
                Vector3 diff = transform.position - col.transform.position;
                force += diff.normalized * separationForce;
            }
        }

        // --- Height Stabilization ---
        float heightError = targetPosition.y - transform.position.y;
        force += Vector3.up * heightError * hoverForce;

        // --- Damping ---
        force += -rb.linearVelocity * damping;

        rb.AddForce(force);

        // --- Clamp Speed ---
        if (rb.linearVelocity.magnitude > maxSpeed)
            rb.linearVelocity = rb.linearVelocity.normalized * maxSpeed;
    }

    public void FailDrone()
    {
        isFailed = true;
        rb.useGravity = true;
    }
}