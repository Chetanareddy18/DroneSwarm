using UnityEngine;
using System.Collections.Generic;

public class SwarmManager : MonoBehaviour
{
    public GameObject dronePrefab;
    public int droneCount = 30;
    public float formationHeight = 20f;
    public float spacing = 4f;

    private List<DroneSwarm> drones = new List<DroneSwarm>();
    private int formationIndex = 0;

    void Start()
    {
        SpawnDrones();
        Invoke("StartShow", 3f);
    }

    void SpawnDrones()
    {
        for (int i = 0; i < droneCount; i++)
        {
            Vector3 spawnPos = new Vector3(
                Random.Range(-15, 15),
                0,
                Random.Range(-15, 15)
            );

            GameObject obj = Instantiate(dronePrefab, spawnPos, Quaternion.identity);
            DroneSwarm drone = obj.GetComponent<DroneSwarm>();
            drone.targetPosition = spawnPos;

            drones.Add(drone);
        }
    }

    void StartShow()
    {
        FormSquare();

        Invoke("FailTwoDrones", 6f);

        // Change formation every 8 seconds continuously
        InvokeRepeating("NextFormation", 10f, 8f);
    }

    void FailTwoDrones()
    {
        if (drones.Count < 3) return;

        drones[0].FailDrone();
        drones[1].FailDrone();

        drones.RemoveAt(0);
        drones.RemoveAt(0);
    }

    void NextFormation()
    {
        formationIndex++;

        if (formationIndex % 3 == 0)
            FormSquare();
        else if (formationIndex % 3 == 1)
            FormCircle();
        else
            FormTriangle();
    }

    // ---------------- SQUARE ----------------
    void FormSquare()
    {
        int count = drones.Count;

        int gridSize = Mathf.CeilToInt(Mathf.Sqrt(count));
        float offset = (gridSize - 1) * spacing / 2f;

        int index = 0;

        for (int x = 0; x < gridSize; x++)
        {
            for (int z = 0; z < gridSize; z++)
            {
                if (index >= count) return;

                Vector3 target = new Vector3(
                    x * spacing - offset,
                    formationHeight,
                    z * spacing - offset
                );

                drones[index].targetPosition = target;
                index++;
            }
        }
    }

    // ---------------- CIRCLE ----------------
    void FormCircle()
    {
        int count = drones.Count;

        // Radius auto scales with drone count
        float radius = Mathf.Sqrt(count) * 4f;

        for (int i = 0; i < count; i++)
        {
            float angle = i * Mathf.PI * 2f / count;

            Vector3 target = new Vector3(
                Mathf.Cos(angle) * radius,
                formationHeight,
                Mathf.Sin(angle) * radius
            );

            drones[i].targetPosition = target;
        }
    }

    // ---------------- TRIANGLE ----------------
    void FormTriangle()
    {
        int count = drones.Count;

        float size = Mathf.Sqrt(count) * 6f;

        for (int i = 0; i < count; i++)
        {
            float t = (float)i / count;

            float x = 0;
            float z = 0;

            if (t < 0.33f)
            {
                float localT = t / 0.33f;
                x = Mathf.Lerp(-size / 2, size / 2, localT);
                z = -size / 2;
            }
            else if (t < 0.66f)
            {
                float localT = (t - 0.33f) / 0.33f;
                x = size / 2;
                z = Mathf.Lerp(-size / 2, size / 2, localT);
            }
            else
            {
                float localT = (t - 0.66f) / 0.34f;
                x = Mathf.Lerp(size / 2, -size / 2, localT);
                z = size / 2;
            }

            drones[i].targetPosition = new Vector3(x, formationHeight, z);
        }
    }
}